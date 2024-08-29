"""Convert comments in a csv into a markdown formatted response suitable for posting

Initial version based on ChatGPT 4o, and then heavily modified.

Once I get the output md, I got through md -> obsidian -> export better pdf
(there are some issues with e.g. direct pandoc conversion to pdf,
so this seemed the path of least resistance, and is good enough)

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import copy
import csv
import re
from pathlib import Path


def extract_sorting_value(line: str) -> int | None:
    """Extract the sorting value from the line column."""
    try:
        parts = re.split(r"\s|-", line)
        return int(parts[0])
    except ValueError:
        return None


def read_csv(file_path: Path) -> list[dict[str, str]]:
    """Read the CSV file and return a list of dictionaries representing the comments."""
    comments = []
    with file_path.open(mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            comments.append(
                {
                    "Line": row["Line"],
                    "Comment": row["Comment"],
                    "Commenter": row["Commenter"],
                    "Comment Type": row["Comment type"],
                    "Response": row["Response"],
                }
            )
    return comments


def sort_comments_by_commenter(comments: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort the comments by Commenter, Comment Type, and Line."""
    comment_type_order = {
        "General physics message": 0,
        "Analysis": 1,
        "Text: general": 2,
        "Text: detailed": 3,
        "Figures and Tables": 4,
        "Bibliography: general": 5,
        "Bibliography": 6,
    }
    comments.sort(
        key=lambda x: (
            x["Commenter"],
            comment_type_order[x["Comment Type"]],
            extract_sorting_value(x["Line"]) if extract_sorting_value(x["Line"]) is not None else float("inf"),
        )
    )
    return comments


def sort_comments_solely_by_line(comments: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort the comments by Commenter, Comment Type, and Line."""
    comments.sort(
        key=lambda x: (
            extract_sorting_value(x["Line"]) if extract_sorting_value(x["Line"]) is not None else float("inf")
        )
    )
    return comments


def format_comment(comment: str) -> str:
    """Format the comment to ensure new lines are quoted correctly in markdown."""
    return comment.replace("\n", "\n>")


def generate_markdown(
    comments: list[dict[str, str]],
    comments_sorted_by_line: list[dict[str, str]],
    output_filename: Path | str = Path("responses.md"),
) -> str:
    """Generate the markdown output from the sorted comments."""
    # Validation
    output_filename = Path(output_filename)

    current_commenter = ""
    current_comment_type = ""

    with output_filename.open("w") as output:
        # Table of contents

        output.write("# Hardest kt, CR1 Responses\n\n")
        output.write("**INSERT GENERAL COMMENT + THANKS**\n\n")
        output.write("Jump to comments:\n\n")
        # NOTE: We want an ordered set, so we'll use dict keys
        people = {comment["Commenter"]: True for comment in comments}
        for person in people:
            output.write(f"- [{person}](#{person.replace(' ', '-')})\n")
        output.write("- [Comments by line](#Comments-by-line)\n\n")

        output.write("-------")

        # Comments sorted solely by line
        for comment in comments:
            if comment["Commenter"] != current_commenter:
                current_commenter = comment["Commenter"]
                output.write("-------\n")
                output.write(f"\n## {current_commenter}\n\n")

            if comment["Comment Type"] != current_comment_type:
                current_comment_type = comment["Comment Type"]
                output.write(f"\n### {current_comment_type}\n\n")

            line_sorting_value = extract_sorting_value(comment["Line"])
            L_prefix = "L" if line_sorting_value is not None and line_sorting_value < 10000 else ""
            # if "Ref" in comment["Line"]:
            #    print(f"Ref: {comment['Line']}")
            output.write(
                f"> {L_prefix}{comment['Line']}: {format_comment(comment['Comment'])}\n\n<span style='color:red'>{comment['Response']}</span>\n\n"
            )

        output.write("-------")
        output.write("\n\n## Comments by line\n\n")

        # Add a section for comments sorted solely by line
        for comment in comments_sorted_by_line:
            line_sorting_value = extract_sorting_value(comment["Line"])
            L_prefix = "L" if line_sorting_value is not None and line_sorting_value < 10000 else ""
            output.write(
                f"> {L_prefix}{comment['Line']}, {comment['Commenter']}: {format_comment(comment['Comment'])}\n\n<span style='color:red'>{comment['Response']}</span>\n\n"
            )


def main(file_path: str | Path) -> None:
    """Main function to read, sort, and generate markdown from the CSV file."""
    path = Path(file_path)
    comments = read_csv(path)
    sorted_comments = sort_comments_by_commenter(comments)
    comments_sorted_by_line = sort_comments_solely_by_line(copy.deepcopy(comments))
    generate_markdown(sorted_comments, comments_sorted_by_line=comments_sorted_by_line)


if __name__ == "__main__":
    main(Path("alice-hardest-kt-paper-comments-collaboration-round-1-sheet-1.csv"))
