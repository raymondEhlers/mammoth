"""Started as bug repro, but I realized that my assumptions were wrong.

See the note at the bottom

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import awkward as ak
import vector

vector.register_awkward()

particles = ak.from_parquet("cannot_broadcast_nested_list_particles.parquet")["particles"]

# All fine
left, right = ak.unzip(ak.combinations(particles, 2))
distances = left.deltaR(right)

# If I mask the initial particles, also fine.
selected_particles = particles[particles["source_index"] >= 1000]
left2, right2 = ak.unzip(ak.combinations(selected_particles, 2))
distances2 = left2.deltaR(right2)

# However, masking on the combinations doesn't work
left_mask = left["source_index"] >= 1000
right_mask = right["source_index"] >= 1000

# NOTE: For future me, this isn't actually a bug! Part of combinations is that it's broadcasting
#       left and right to be the same length. When we mask left and right, they end up being different
#       lengths, which causes the deltaR calculation to fail.
#       If I wanted to mask left and right, I need to figure out their combined mask, and then apply
#       that to the distances! This is demonstrated below:
# Raises here.
# distances3 = left[left_mask].deltaR(right[right_mask])
# Correct way to do it:
combined_mask = (left["source_index"] >= 1000) & (right["source_index"] >= 1000)
distances4 = distances[combined_mask]
assert distances4.to_list() == distances2.to_list()

print("Success!")
