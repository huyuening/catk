import unittest

from src.womd_labeling.agent_size_classification import (
    AASHTO_PASSENGER_CAR_LENGTH_M,
    FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS,
    MOTORCYCLE_BOX_MAX_LENGTH_M,
    MOTORCYCLE_BOX_MAX_WIDTH_M,
    NHTSA_FOUR_FEET_NINE_INCHES_M,
    classify_agent_dimensions,
    extract_agent_size_records,
)
from src.womd_labeling.proto import scenario_pb2


class AgentSizeClassificationTest(unittest.TestCase):
    def test_vehicle_uses_19_foot_length_boundary(self):
        small = classify_agent_dimensions(
            1, AASHTO_PASSENGER_CAR_LENGTH_M, 2.0, 1.7
        )
        large = classify_agent_dimensions(
            1, AASHTO_PASSENGER_CAR_LENGTH_M + 0.01, 2.0, 1.7
        )

        self.assertEqual(small.size_class, "SMALL_VEHICLE_PROXY")
        self.assertEqual(large.size_class, "LARGE_VEHICLE_PROXY")

    def test_vehicle_motorcycle_proxy_precedes_length_classes(self):
        motorcycle = classify_agent_dimensions(
            1,
            MOTORCYCLE_BOX_MAX_LENGTH_M,
            MOTORCYCLE_BOX_MAX_WIDTH_M,
            1.7,
        )
        narrow_car = classify_agent_dimensions(
            1,
            MOTORCYCLE_BOX_MAX_LENGTH_M + 0.01,
            MOTORCYCLE_BOX_MAX_WIDTH_M,
            1.7,
        )

        self.assertEqual(motorcycle.size_class, "MOTORCYCLE_PROXY")
        self.assertEqual(narrow_car.size_class, "SMALL_VEHICLE_PROXY")

    def test_cyclist_uses_fhwa_speed_proxy(self):
        bicycle = classify_agent_dimensions(
            3,
            1.8,
            0.8,
            1.8,
            speed_mps=FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS - 0.01,
        )
        ebike = classify_agent_dimensions(
            3,
            1.8,
            0.8,
            1.8,
            speed_mps=FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS,
        )

        self.assertEqual(bicycle.size_class, "BICYCLE_PROXY")
        self.assertEqual(ebike.size_class, "E_BIKE_PROXY")

    def test_pedestrian_uses_four_foot_nine_height_proxy(self):
        child = classify_agent_dimensions(
            2, 0.8, 0.7, NHTSA_FOUR_FEET_NINE_INCHES_M - 0.01
        )
        adult = classify_agent_dimensions(
            2, 0.8, 0.7, NHTSA_FOUR_FEET_NINE_INCHES_M
        )

        self.assertEqual(child.size_class, "CHILD_PEDESTRIAN_PROXY")
        self.assertEqual(adult.size_class, "ADULT_PEDESTRIAN_PROXY")

    def test_extracts_only_requested_valid_frame(self):
        scenario = scenario_pb2.Scenario()
        scenario.sdc_track_index = 0
        vehicle = scenario.tracks.add()
        vehicle.id = 7
        vehicle.object_type = 1
        for frame_index in range(11):
            state = vehicle.states.add()
            state.length = 4.0 + frame_index * 0.1
            state.width = 2.0
            state.height = 1.6
            state.velocity_x = 3.0
            state.velocity_y = 4.0
            state.valid = frame_index == 10

        invalid = scenario.tracks.add()
        invalid.id = 8
        invalid.object_type = 2
        invalid.states.add().valid = False

        records, diagnostics = extract_agent_size_records(scenario, 10)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["frame_number"], 11)
        self.assertAlmostEqual(records[0]["length_m"], 5.0)
        self.assertAlmostEqual(records[0]["speed_mps"], 5.0)
        self.assertEqual(records[0]["size_class"], "SMALL_VEHICLE_PROXY")
        self.assertEqual(diagnostics["valid_state"], 1)
        self.assertEqual(diagnostics["missing_state"], 1)

    def test_rejects_nonpositive_dimensions(self):
        label = classify_agent_dimensions(1, 0.0, 2.0, 1.5)
        self.assertEqual(label.size_class, "INVALID_DIMENSIONS")
        self.assertFalse(label.supported)


if __name__ == "__main__":
    unittest.main()
