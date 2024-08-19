"""Update citations as needed for the kt paper

Given a .bib file, this script will update the citations in the .tex file.
Namely, it will retrieve the citation from InspireHEP, and then add missing fields
into the existing citations from the .bib file, writing to a new file called `<name>-updated.bib`

(Written with some help from Claude :-) )

.. codeauthor: Raymond Ehlers <raymond.ehlers@cern>, LBL/UCB
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import bibtexparser
import requests

from mammoth import helpers

logger = logging.getLogger(__name__)


def get_inspire_bibtex(citation_key: str) -> str | None:
    url = f"https://inspirehep.net/api/literature?q=texkey:{citation_key}&format=bibtex"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return None


def merge_entries(original_entry: dict[str, str], inspire_entry: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    # Fields to update or add
    # NOTE: RJE:
    #     If we don't want to modify fields (e.g. titles that we've updated by hand),
    #     it's better to omit them here. Otherwise, we want to update all fields listed
    #     here to e.g. update the journal from arXiv to the proper journal name + values.
    fields_to_update = ["volume", "number", "pages", "year", "doi", "journal", "eprint"]

    # Update the requested fields if available
    updated_keys = []
    for field in fields_to_update:
        if field in inspire_entry:
            original_entry[field] = inspire_entry[field]
            updated_keys.append(field)

    return original_entry, updated_keys


def update_bib_file(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    with input_path.open("r") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    updated_entries = []

    with helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(bib_database.entries), description="Processing entries...")
        for entry in bib_database.entries:
            citation_key = entry["ID"]
            inspire_bibtex = get_inspire_bibtex(citation_key)

            updated_keys = []
            if inspire_bibtex:
                inspire_db = bibtexparser.loads(inspire_bibtex)
                if inspire_db.entries:
                    inspire_entry = inspire_db.entries[0]
                    updated_entry, updated_keys = merge_entries(entry.copy(), inspire_entry)
                    updated_entries.append(updated_entry)
            else:
                logger.warning(f"InspireHEP entry not found for citation key: {citation_key}")
                updated_entries.append(entry)
            logger.info(f'Entry: "{citation_key}", updated fields: {updated_keys}')
            progress.update(track_results, advance=1)
            # Don't hit InspireHEP too hard
            time.sleep(2)

    bib_database.entries = updated_entries

    with output_path.open("w") as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)


def main(input_file: Path = Path("input.bib")) -> None:
    helpers.setup_logging()

    output_file = input_file.with_name(f"{input_file.stem}-updated{input_file.suffix}")
    update_bib_file(input_file, output_file)
    logger.info(f"Updated BibTeX file saved as {output_file}")


if __name__ == "__main__":
    main(input_file=Path("bibliography_kt_paper_2024_08_19-reference-updated.bib"))
