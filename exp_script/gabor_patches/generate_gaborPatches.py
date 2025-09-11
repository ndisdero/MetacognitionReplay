from psychopy import visual, core, event, gui, prefs
import psychopy
import numpy as np
from gabor_patches import utils
import copy
from gabor_patches.constant import constant as ct
import random as rd
# Create the gabor patches
# Generate the 15 Gabor patches with different orientations


def arc_vertices(radius, start, end, edges=190):
    """Generates vertices for an arc."""
    theta = np.linspace(start, end, edges)
    theta = np.deg2rad(-theta + 90)
    return np.column_stack([radius*np.cos(theta), radius*np.sin(theta)]).tolist()

def generate_gabor_patches(win, reference, direction, distance_to_bound):
    """ Generates Gabor patches for the experiment.
    Reference determines the angle of the reference that the mean oirentation has to be compared to
    If direction is 1 then the orientation is at right (blue), if -1 then left
    """

    # Arc of answer
    arc_left = visual.ShapeStim(
        win,
        closeShape=False,          # <- keeps it as a line (no closing/radials)
        lineColor='orange',
        lineWidth=4,
        fillColor=None,
    )
    arc_left_sym = copy.copy(arc_left)

    arc_right = visual.ShapeStim(
        win,
        closeShape=False,          # <- keeps it as a line (no closing/radials)
        lineColor='blue',
        lineWidth=4,
        fillColor=None,
    )
    arc_right_sym = copy.copy(arc_right)

        
    # Defines the angles that will be displayed to answer
    arc_left.vertices = arc_vertices(radius=ct.DIAGONAL_TO_CENTER_ARC, start=reference-45, end=reference)
    arc_left_sym.vertices = arc_vertices(radius=ct.DIAGONAL_TO_CENTER_ARC, start=reference-45+180, end=reference+180)
    
    arc_right.vertices = arc_vertices(radius=ct.DIAGONAL_TO_CENTER_ARC, start=reference, end=reference+45)
    arc_right_sym.vertices = arc_vertices(radius=ct.DIAGONAL_TO_CENTER_ARC, start=reference+180, end=reference+45+180)
    # Store things to show in a list
    to_show_arc = [arc_left, arc_left_sym, arc_right, arc_right_sym]
    
    orientations = utils.generate_orientation_patches(ct.NUM_PATCHES, reference+distance_to_bound*direction, sd=5)
    positions = utils.generate_position_patches(ct.NUM_PATCHES, ct.DIAGONAL_TO_CENTER, ct.DISTANCE_MIN_BETW_GABORS)
    
    # Also for the gabor patches
    to_show_gabor = []
    for i in range(ct.NUM_PATCHES):
        stim = psychopy.visual.GratingStim(win, ori=orientations[i], sf=0.15, 
                                        contrast = 3,
                                        size=(ct.GABOR_SIZE, ct.GABOR_SIZE),
                                        pos=positions[i],
                                        color='white', mask='gauss')
        to_show_gabor.append(stim)

    return to_show_gabor, to_show_arc

if __name__ == "__main__":
    win = visual.Window([800, 600], units='pix', fullscr=False, color = "gray")
    rt_clock = core.Clock()
    key_list = ["d", "f"]


    for i in range(ct.NUMBER_TRIALS):
        direction=rd.choice([-1, 1])
        print(direction)
        to_show = generate_gabor_patches(win,
                                        reference=np.random.randint(0, 360),
                                        direction=direction, distance_to_bound=10)
        # Display the Gabor patches
        for stim in to_show[0]:
            stim.draw()
        # And the arcs
        for stim in to_show[1]:
            stim.draw()

        win.flip()
        # Wait for a key press
        rt_clock.reset()
        core.wait(.25)
        for stim in to_show[1]:
            stim.draw()

        event.clearEvents()
        while True: # Wait for participant's answer
            key = event.getKeys(keyList=key_list, timeStamped=rt_clock)
            if key != []:
                info_answer = key[0] # take the first response
                answer = info_answer[0]
                rt = info_answer[1]
                break
        win.flip()
        if answer == "f" and direction == 1 or answer == "d" and direction == -1:
            print("CORRECT")
        else:
            print("INCORRECT")

        win.flip()
            # And the arcs
        core.wait(1)

