import numpy as np
class PostureDefinition:
    def __init__(self, name, points, threshold, ideal_range):
        self.name = name
        self.points = points  # List of points as (landmark indices)
        self.threshold = threshold
        self.ideal_range = ideal_range

    def get_points(self, landmarks):
        # Extract the specific landmarks for the posture based on the points indices
        return [landmarks[i] for i in self.points]


class PostureDefinitions:
    def __init__(self):
        # Define each posture with its associated points and thresholds
        self.postures = {
            "Forward Head Posture": PostureDefinition("Forward Head Posture", [7, 11, 23], 45, (0, 20)), #1
            # "Rounded Shoulders": PostureDefinition("Rounded Shoulders", [11, 12, 23], 30, (0, 15)), #2
            # "Slouching or Kyphotic Posture": PostureDefinition("Slouching or Kyphotic Posture", [0, 12, 24], 40, (0, 20)), #3
            "Anterior Pelvic Tilt": PostureDefinition("Anterior Pelvic Tilt", [23, 25, 27], 10, (-5, 5)), #4
            "Posterior Pelvic Tilt": PostureDefinition("Posterior Pelvic Tilt", [23, 25, 27], -10, (-5, 5)), #5
            # "Uneven Shoulders": PostureDefinition("Uneven Shoulders", [11, 12, 24], 5, (-5, 5)),#6
            # "Uneven Hips": PostureDefinition("Uneven Hips", [23, 24, 26], 5, (-5, 5)), #7
            # "Lordosis": PostureDefinition("Lordosis", [24, 26, 28], 30, (10, 20)), #8
            "Knee Valgus (Knock Knees)": PostureDefinition("Knee Valgus (Knock Knees)", [23, 25, 27], 170, (170, 180)), #9
            "Genu Varum (Bow Legs)": PostureDefinition("Genu Varum (Bow Legs)", [23, 25, 27], 180, (170, 180)), #10
            # "Foot Pronation": PostureDefinition("Foot Pronation", [27, 31], 10, (0, 10)), # inward tilt #11
            # "Foot Supination": PostureDefinition("Foot Supination", [27, 31], -10, (-10, 0)), # outward tilt #12
            # "Asymmetrical Weight Distribution": PostureDefinition("Asymmetrical Weight Distribution", [11, 23], 5, (-5, 5)), #13
            "Text Neck (Cervical Spine Strain)": PostureDefinition("Text Neck", [0, 11, 23], 45, (0, 15)), #14
            "Gait Issues (Stride Consistency)": PostureDefinition("Gait Issues", [23, 25, 27], 0, (170, 180)), # Placeholder #15
            "Trendelenburg Gait (Lateral Hip Shift)": PostureDefinition("Trendelenburg Gait", [23, 24, 26], 5, (-5, 5)), #16 
            "Excessive Forward Leaning": PostureDefinition("Excessive Forward Leaning", [11, 23, 25], 20, (0, 10)), #17
        }
    def get_posture(self, name):
        return self.postures.get(name)

def angle_calc(p1, p2, p3):
    p1 = np.array(p1[:2])
    p2 = np.array(p2[:2])
    p3 = np.array(p3[:2])
    radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle %= 360
    return angle

def main():
    pos = PostureDefinitions()
    print(pos.get_posture("Forward Head Posture"))

if __name__ == "__main__":
    main()