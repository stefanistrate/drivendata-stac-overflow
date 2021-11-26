"""Useful colormaps.

Colormaps for JRC Global Surface Water data are based on the official user
guide:
https://storage.cloud.google.com/global-surface-water/downloads_ancillary/DataUsersGuidev2020.pdf
"""

from matplotlib.colors import LinearSegmentedColormap

CM_JRC_CHANGE = LinearSegmentedColormap.from_list("jrc_change", [
    (0 / 255, "#FF0000"),
    (100 / 255, "#000000"),
    (200 / 255, "#00FF00"),
    (253 / 255, "#FFFFFF"),
    (254 / 255, "#888888"),
    (255 / 255, "#CCCCCC"),
])

CM_JRC_EXTENT = LinearSegmentedColormap.from_list("jrc_extent", [
    (0 / 255, "#FFFFFF"),
    (1 / 255, "#6666FF"),
    (255 / 255, "#CCCCCC"),
])

CM_JRC_OCCURRENCE = LinearSegmentedColormap.from_list(
    "jrc_occurrence",
    [
        (0 / 255, "#FFFFFF"),
        (1 / 255, (1.0, 0.0, 0.0, 0.01)),  # FF0000 with 1% opacity
        (100 / 255, (0.0, 0.0, 1.0, 1.0)),  # 0000FF with 100% opacity
        (255 / 255, "#CCCCCC"),
    ])

CM_JRC_RECURRENCE = LinearSegmentedColormap.from_list("jrc_recurrence", [
    (0 / 255, "#FFFFFF"),
    (1 / 255, "#FF7F27"),
    (100 / 255, "#99D9EA"),
    (255 / 255, "#CCCCCC"),
])

CM_JRC_SEASONALITY = LinearSegmentedColormap.from_list("jrc_seasonality", [
    (0 / 255, "#FFFFFF"),
    (1 / 255, "#99D9EA"),
    (12 / 255, "#0000AA"),
    (255 / 255, "#CCCCCC"),
])

CM_JRC_TRANSITIONS = LinearSegmentedColormap.from_list("jrc_transitions", [
    (0 / 255, "#FFFFFF"),
    (1 / 255, "#0000FF"),
    (2 / 255, "#22B14C"),
    (3 / 255, "#D1102D"),
    (4 / 255, "#99D9EA"),
    (5 / 255, "#B5E61D"),
    (6 / 255, "#E6A1AA"),
    (7 / 255, "#FF7F27"),
    (8 / 255, "#FFC90E"),
    (9 / 255, "#7F7F7F"),
    (10 / 255, "#C3C3C3"),
    (255 / 255, "#CCCCCC"),
])

CM_WATER_LABEL = LinearSegmentedColormap.from_list("water_label", [
    (0 / 255, "#FFFFFF"),
    (1 / 255, "#0000FF"),
    (255 / 255, "#CCCCCC"),
])
