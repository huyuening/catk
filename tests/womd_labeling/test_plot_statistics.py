import gzip
import json
from pathlib import Path
import tempfile
import unittest

from matplotlib.path import Path as MatplotlibPath

from src.womd_labeling import plot_statistics as summary_plot


class PlotStatisticsSummaryTest(unittest.TestCase):
    def test_compact_count_uses_one_decimal_and_metric_suffixes(self):
        cases = (
            (12, "12"),
            (999, "999"),
            (1_000, "1.0 K"),
            (125_000, "125.0 K"),
            (1_250_000, "1.2 M"),
            (1_260_000_000, "1.3 B"),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(
                    summary_plot._format_compact_count(value),
                    expected,
                )

    def test_recount_road_hierarchy_uses_environment_and_region_fields(self):
        frames = (
            self._frame("FREEWAY", "ROAD_SEGMENT", "FREEWAY_MAINLINE"),
            self._frame("FREEWAY", "ROAD_SEGMENT", "FREEWAY_RAMP"),
            self._frame("URBAN_STREET", "INTERSECTION"),
            self._frame("URBAN_STREET", "ROAD_SEGMENT"),
            self._frame("PARKING_LOT", "ROAD_SEGMENT"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for index, frame in enumerate(frames):
                    json.dump(
                        {
                            "scenario_id": f"scenario-{index}",
                            "current_time_index": 10,
                            "ego_frames": [frame],
                        },
                        stream,
                    )
                    stream.write("\n")

            result = summary_plot.recount_road_hierarchy(path, frame_index=10)

        self.assertEqual(
            dict(result["top_counts"]),
            {"Freeway": 2, "Urban road": 3},
        )
        self.assertEqual(
            dict(result["child_counts"]["Freeway"]),
            {"Mainline": 1, "Ramp": 1},
        )
        self.assertEqual(
            dict(result["child_counts"]["Urban road"]),
            {
                "Intersection": 1,
                "Road segment": 1,
                "Parking lot": 1,
            },
        )
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["unknown"], 0)

    def test_recount_road_hierarchy_combines_multiple_annotation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, frame in enumerate(
                (
                    self._frame("FREEWAY", "ROAD_SEGMENT", "FREEWAY_MAINLINE"),
                    self._frame("URBAN_STREET", "INTERSECTION"),
                )
            ):
                path = Path(directory) / f"annotations-{index}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "scenario_id": f"scenario-{index}",
                            "current_time_index": 10,
                            "ego_frames": [frame],
                        },
                        stream,
                    )
                    stream.write("\n")
                paths.append(path)

            result = summary_plot.recount_road_hierarchy(paths, frame_index=10)

        self.assertEqual(result["rows"], 2)
        self.assertEqual(
            dict(result["top_counts"]),
            {"Freeway": 1, "Urban road": 1},
        )

    def test_agent_hierarchy_merges_cyclists_without_moving_motorcyclists(
        self,
    ):
        sizes = {
            "counts": {
                "LARGE_VEHICLE_PROXY": 714,
                "SMALL_VEHICLE_PROXY": 14_995,
                "MOTORCYCLE_PROXY": 26,
                "E_BIKE_PROXY": 9,
                "BICYCLE_PROXY": 100,
                "ADULT_PEDESTRIAN_PROXY": 1_505,
                "CHILD_PEDESTRIAN_PROXY": 62,
            }
        }

        groups = summary_plot.build_agent_hierarchy(sizes)

        self.assertEqual(
            groups[0],
            {
                "label": "Vehicle",
                "count": 15_735,
                "children": {
                    "Large vehicle": 714,
                    "Small vehicle": 14_995,
                    "Motorcyclist": 26,
                },
            },
        )
        self.assertEqual(
            groups[1],
            {"label": "Cyclist", "count": 109, "children": {}},
        )
        self.assertEqual(
            groups[2],
            {
                "label": "Pedestrian",
                "count": 1_567,
                "children": {"Adult": 1_505, "Child": 62},
            },
        )

    def test_sunburst_layout_preserves_true_angles_and_terminal_leaf(self):
        groups = (
            {
                "label": "Freeway",
                "count": 6,
                "children": {"Mainline": 5, "Ramp": 1},
            },
            {
                "label": "Urban road",
                "count": 490,
                "children": {
                    "Intersection": 334,
                    "Road segment": 142,
                    "Parking lot": 14,
                },
            },
        )

        nodes = summary_plot._build_sunburst_nodes(groups)
        by_label = {node["label"]: node for node in nodes}

        self.assertAlmostEqual(
            by_label["Freeway"]["start_angle"]
            - by_label["Freeway"]["end_angle"],
            360.0 * 6 / 496,
        )
        self.assertAlmostEqual(
            by_label["Mainline"]["start_angle"]
            - by_label["Mainline"]["end_angle"],
            360.0 * 5 / 496,
        )
        self.assertEqual(by_label["Freeway"]["depth"], 0)
        self.assertEqual(by_label["Mainline"]["depth"], 1)

        agent_nodes = summary_plot._build_sunburst_nodes(
            (
                {
                    "label": "Vehicle",
                    "count": 10,
                    "children": {"Small vehicle": 10},
                },
                {"label": "Cyclist", "count": 1, "children": {}},
            )
        )
        cyclist_nodes = [
            node for node in agent_nodes if node["label"] == "Cyclist"
        ]
        self.assertEqual(len(cyclist_nodes), 1)
        self.assertTrue(cyclist_nodes[0]["terminal"])

    def test_action_panel_uses_compact_labels_inside_heatmap(self):
        actions = {
            "counts": {
                ("TYPE_VEHICLE", 2): 1_250_000,
                ("TYPE_CYCLIST", 3): 125_000,
                ("TYPE_PEDESTRIAN", 5): 12,
            }
        }
        figure, axis = summary_plot.plt.subplots()
        self.addCleanup(summary_plot.plt.close, figure)

        summary_plot.draw_action_panel(axis, actions)

        labels = {text.get_text() for text in axis.texts}
        self.assertIn("1.2 M", labels)
        self.assertIn("125.0 K", labels)
        self.assertIn("12", labels)
        self.assertNotIn("1,250,000", labels)

    def test_sunburst_renderer_draws_terminal_leaf_once_with_callout(self):
        groups = (
            {
                "label": "Vehicle",
                "count": 99,
                "children": {"Small vehicle": 99},
            },
            {"label": "Cyclist", "count": 1, "children": {}},
        )
        palette = {
            "Vehicle": "#B84A3C",
            "Small vehicle": "#D98978",
            "Cyclist": "#D39A27",
        }
        figure, axis = summary_plot.plt.subplots()
        self.addCleanup(summary_plot.plt.close, figure)

        rows = summary_plot.draw_sunburst_panel(
            axis,
            groups,
            palette,
            panel_label="b",
            panel_key="b",
            unit="agent",
        )

        wedges = [
            patch
            for patch in axis.patches
            if isinstance(patch, summary_plot.Wedge)
        ]
        self.assertEqual(len(wedges), 3)
        self.assertEqual(
            [row["label"] for row in rows],
            ["Vehicle", "Small vehicle", "Cyclist"],
        )
        self.assertEqual(
            sum(
                text.get_text() == "Cyclist\n1"
                for text in axis.texts
            ),
            1,
        )
        self.assertGreaterEqual(len(axis.lines), 1)

    def test_sunburst_renderer_uses_compact_count_labels(self):
        groups = (
            {
                "label": "Vehicle",
                "count": 15_735,
                "children": {
                    "Small vehicle": 14_995,
                    "Motorcyclist": 740,
                },
            },
        )
        palette = {
            "Vehicle": "#B84A3C",
            "Small vehicle": "#D98978",
            "Motorcyclist": "#F2B9A9",
        }
        figure, axis = summary_plot.plt.subplots()
        self.addCleanup(summary_plot.plt.close, figure)

        summary_plot.draw_sunburst_panel(
            axis,
            groups,
            palette,
            panel_label="b",
            panel_key="b",
            unit="agent",
        )

        labels = {text.get_text() for text in axis.texts}
        self.assertIn("Vehicle\n15.7 K", labels)
        self.assertIn("Small vehicle\n15.0 K", labels)
        self.assertIn("Motorcyclist\n740", labels)
        self.assertNotIn("Vehicle\n15,735", labels)

    def test_small_top_level_agent_labels_are_centered_inside_inner_ring(
        self,
    ):
        sizes = {
            "counts": {
                "LARGE_VEHICLE_PROXY": 714,
                "SMALL_VEHICLE_PROXY": 14_995,
                "MOTORCYCLE_PROXY": 26,
                "E_BIKE_PROXY": 9,
                "BICYCLE_PROXY": 100,
                "ADULT_PEDESTRIAN_PROXY": 1_505,
                "CHILD_PEDESTRIAN_PROXY": 62,
            }
        }
        figure, axis = self._publication_top_axis(column=1)
        self.addCleanup(summary_plot.plt.close, figure)

        summary_plot.draw_size_panel(axis, sizes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = {text.get_text(): text for text in axis.texts}
        target_texts = (
            texts["Pedestrian\n1.6 K"],
            texts["Cyclist\n109"],
        )

        for text in target_texts:
            x, y = text.get_position()
            radius = (x**2 + y**2) ** 0.5
            self.assertGreater(radius, summary_plot.SUNBURST_HOLE_RADIUS)
            self.assertLess(radius, summary_plot.SUNBURST_PARENT_RADIUS)
            self.assertEqual(text._get_multialignment(), "center")

        pedestrian_box = target_texts[0].get_window_extent(renderer)
        cyclist_box = target_texts[1].get_window_extent(renderer)
        self.assertFalse(pedestrian_box.overlaps(cyclist_box))
        hole_left_x = axis.transData.transform(
            (-summary_plot.SUNBURST_HOLE_RADIUS, 0.0)
        )[0]
        self.assertLessEqual(cyclist_box.x1, hole_left_x)
        labelled_boxes = [
            (text.get_text(), text.get_window_extent(renderer))
            for text in axis.texts
            if "\n" in text.get_text()
        ]
        for index, (first_label, first_box) in enumerate(labelled_boxes):
            for second_label, second_box in labelled_boxes[index + 1 :]:
                self.assertFalse(
                    first_box.overlaps(second_box),
                    f"{first_label!r} overlaps {second_label!r}",
                )

    def test_all_two_line_sunburst_labels_are_center_aligned(self):
        road_groups = (
            {
                "label": "Freeway",
                "count": 6,
                "children": {"Mainline": 5, "Ramp": 1},
            },
            {
                "label": "Urban road",
                "count": 490,
                "children": {
                    "Intersection": 334,
                    "Road segment": 142,
                    "Parking lot": 14,
                },
            },
        )
        agent_groups = (
            {
                "label": "Vehicle",
                "count": 15_735,
                "children": {
                    "Large vehicle": 714,
                    "Small vehicle": 14_995,
                    "Motorcyclist": 26,
                },
            },
            {"label": "Cyclist", "count": 109, "children": {}},
            {
                "label": "Pedestrian",
                "count": 1_567,
                "children": {"Adult": 1_505, "Child": 62},
            },
        )
        figure, axes = summary_plot.plt.subplots(1, 2)
        self.addCleanup(summary_plot.plt.close, figure)

        summary_plot.draw_sunburst_panel(
            axes[0],
            road_groups,
            summary_plot.ROAD_SUNBURST_PALETTE,
            panel_label="a",
            panel_key="a",
            unit="scene",
        )
        summary_plot.draw_sunburst_panel(
            axes[1],
            agent_groups,
            summary_plot.AGENT_SUNBURST_PALETTE,
            panel_label="b",
            panel_key="b",
            unit="agent",
        )

        for axis in axes:
            two_line_texts = [
                text for text in axis.texts if "\n" in text.get_text()
            ]
            self.assertGreater(len(two_line_texts), 0)
            for text in two_line_texts:
                self.assertEqual(
                    text._get_multialignment(),
                    "center",
                    text.get_text(),
                )

    def test_sunburst_limits_enlarge_chart_without_clipping_labels(self):
        groups = (
            {
                "label": "Freeway",
                "count": 6,
                "children": {"Mainline": 5, "Ramp": 1},
            },
            {
                "label": "Urban road",
                "count": 490,
                "children": {
                    "Intersection": 334,
                    "Road segment": 142,
                    "Parking lot": 14,
                },
            },
        )
        figure, axis = self._publication_top_axis(column=0)
        self.addCleanup(summary_plot.plt.close, figure)

        summary_plot.draw_sunburst_panel(
            axis,
            groups,
            summary_plot.ROAD_SUNBURST_PALETTE,
            panel_label="a",
            panel_key="a",
            unit="scene",
            start_angle=182.2,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()

        self.assertLessEqual(axis.get_xlim()[1] - axis.get_xlim()[0], 3.90)
        self.assertLessEqual(axis.get_ylim()[1] - axis.get_ylim()[0], 3.10)
        figure_box = figure.bbox
        for text in axis.texts:
            if "\n" not in text.get_text():
                continue
            text_box = text.get_window_extent(renderer)
            self.assertGreaterEqual(text_box.x0, figure_box.x0)
            self.assertLessEqual(text_box.x1, figure_box.x1)
            self.assertGreaterEqual(text_box.y0, figure_box.y0)
            self.assertLessEqual(text_box.y1, figure_box.y1)

    def test_sunburst_callout_text_boxes_do_not_overlap(self):
        groups = (
            {
                "label": "Freeway",
                "count": 6,
                "children": {"Mainline": 5, "Ramp": 1},
            },
            {
                "label": "Urban road",
                "count": 490,
                "children": {
                    "Intersection": 334,
                    "Road segment": 142,
                    "Parking lot": 14,
                },
            },
        )
        figure, axis = self._publication_top_axis(column=0)
        self.addCleanup(summary_plot.plt.close, figure)
        summary_plot.draw_sunburst_panel(
            axis,
            groups,
            summary_plot.ROAD_SUNBURST_PALETTE,
            panel_label="a",
            panel_key="a",
            unit="scene",
            start_angle=182.2,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        callout_labels = {
            "Freeway\n6",
            "Mainline\n5",
            "Ramp\n1",
            "Parking lot\n14",
        }
        boxes = [
            text.get_window_extent(renderer)
            for text in axis.texts
            if text.get_text() in callout_labels
        ]

        self.assertEqual(len(boxes), len(callout_labels))
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                self.assertFalse(first.overlaps(second))

    def test_nested_sunburst_parent_and_child_text_boxes_do_not_overlap(self):
        groups = (
            {
                "label": "Vehicle",
                "count": 15_735,
                "children": {
                    "Large vehicle": 714,
                    "Small vehicle": 14_995,
                    "Motorcyclist": 26,
                },
            },
            {"label": "Cyclist", "count": 109, "children": {}},
            {
                "label": "Pedestrian",
                "count": 1_567,
                "children": {"Adult": 1_505, "Child": 62},
            },
        )
        figure, axis = self._publication_top_axis(column=1)
        self.addCleanup(summary_plot.plt.close, figure)
        summary_plot.draw_sunburst_panel(
            axis,
            groups,
            summary_plot.AGENT_SUNBURST_PALETTE,
            panel_label="b",
            panel_key="b",
            unit="agent",
            start_angle=146.5,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = {
            text.get_text(): text.get_window_extent(renderer)
            for text in axis.texts
        }

        self.assertFalse(
            boxes["Pedestrian\n1.6 K"].overlaps(boxes["Adult\n1.5 K"])
        )
        adult_box = boxes["Adult\n1.5 K"]
        for line in axis.lines:
            points = axis.transData.transform(
                list(zip(line.get_xdata(), line.get_ydata()))
            )
            self.assertFalse(
                MatplotlibPath(points).intersects_bbox(
                    adult_box,
                    filled=False,
                )
            )

    @staticmethod
    def _frame(
        environment,
        region_type,
        environment_subtype=None,
    ):
        return {
            "frame_index": 10,
            "valid": True,
            "road_environment": environment,
            "road_environment_subtype": environment_subtype,
            "region_type": region_type,
        }

    @staticmethod
    def _publication_top_axis(column):
        figure = summary_plot.plt.figure(figsize=(16 / 2.54, 13 / 2.54))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(1.28, 0.78),
            hspace=-0.03,
            wspace=0.08,
            left=0.13,
            right=0.94,
            top=0.95,
            bottom=0.12,
        )
        return figure, figure.add_subplot(grid[0, column])


if __name__ == "__main__":
    unittest.main()
