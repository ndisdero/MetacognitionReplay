from psychopy import visual, core, event, gui, prefs
import random
import os
from datetime import datetime
import utils
import gabor_patches.generate_gaborPatches as gabor
import numpy as np
import csv

# Window settings
prefs.general['windowType'] = 'pyglet'
win = visual.Window(
    size=[1920, 1200],
    color='gray',
    units='pix',
    fullscr=False,
    screen=0,
    pos=[0, 0]
)
key_list = ["v", "b"]

# Experiment parameters
N_TRAINING_TRIALS = 20  # number of training trials
TOTAL_BLOCKS = 8  # must be even: half with replay, half without
TRIALS_PER_BLOCK = 20  # per block
STIMULUS_DURATION = 0.25
INTERTRIAL_PAUSE = 1  # fixation cross display time
MENTAL_REPLAY_PAUSE = 1.5  # pause during mental replay instruction
FEEDBACK_TIME = 0.5
FIXATION_CROSS_DURATION = 0.5

# Set relative paths
phase = 0
folder = {0: "code", 1: "pilot", 2: "experimental"}
save_directory = f"./data/{folder[phase]}/"
image_dir = "./images"

# Make sure the data folder exists
os.makedirs(save_directory, exist_ok=True)

# Instruction image setup
image_intro = [os.path.join(image_dir, f) for f in ["Intro1.JPG"]]

image_training1 = [os.path.join(image_dir, f) for f in ["Training1_1.JPG", "Training1_2.JPG", "Training1_3.JPG"]] # instructions displayed before 3 consecutive training phases
image_training2 = [os.path.join(image_dir, f) for f in ["Training2_1.JPG", "Training2_2.JPG"]]
image_training3 = [os.path.join(image_dir, f) for f in ["Training3_1.JPG", "Training3_2.JPG"]]
image_training_end = [os.path.join(image_dir, f) for f in ["Training_end.JPG"]]

image_exp_instructions = [os.path.join(image_dir, f) for f in ["Experiment1.JPG", "Experiment2.JPG", "Experiment3.JPG", "Experiment4.JPG"]]

image_break = [os.path.join(image_dir, "Break.jpg")]
image_end = [os.path.join(image_dir, "End.JPG")]

min_display_time = 2  # seconds before intructions can be skipped

# Escape function
def check_for_escape():
    if event.getKeys(keyList=['escape']):
        win.close()
        core.quit()

# TRIAL PHASE
if TOTAL_BLOCKS % 2 != 0:
    print("TOTAL_BLOCKS must be an even number.")
    core.quit()

# Step 1: Get Participant ID
id_dialog = gui.Dlg(title="Enter Participant ID")
id_dialog.addField("Participant ID:")
id_dialog.show()
if not id_dialog.OK:
    core.quit()
participant_id = str(id_dialog.data[0]).strip()

while not participant_id:
    print("Participant ID cannot be empty.")
    participant_id = str(gui.Dlg(title="Enter Participant ID").addField("Participant ID:").show()[0]).strip()
    if not participant_id:
        core.quit()
participant_id_clean = participant_id.replace(":", "_").replace("/", "_").replace("\\", "_").strip()
random_seed = sum(ord(c) for c in participant_id_clean) # generate random seed
random.seed(random_seed)

# Collect reaction time
rt_clock = core.Clock()

# Sorting participants into 8 conditions
# handle numeric and non-numeric participant ids robustly
try:
    participant_id_num = int(participant_id_clean)  # participant ID must be numeric for direct numeric mapping
    group_id = participant_id_num % 4  # assign participant to 1 of 8 groups (0–7)
except ValueError:
    # fallback: use sum of ordinals (consistent with the seed)
    group_id = sum(ord(c) for c in participant_id_clean) % 4

# Factor 1: starting condition
start_with_replay = (group_id % 2 == 0)  # even groups: replay first, odd groups: non-replay first

# Factor 2: hand assignment
left_for_confidence = (group_id // 2) % 2 == 0  # groups 0–3: left hand = confidence; groups 4–7: left hand = vividness

# Create block order
half_blocks = TOTAL_BLOCKS // 2
if start_with_replay:
    block_order = ["with_mental_replay"]*half_blocks + ["without_mental_replay"]*half_blocks
else:
    block_order = ["without_mental_replay"]*half_blocks + ["with_mental_replay"]*half_blocks

test_directions = []

print(f"Assigned to group {group_id}: "
      f"{'Replay first' if start_with_replay else 'Non-replay first'}, "
      f"{'Left=confidence' if left_for_confidence else 'Left=vividness'}")

# Save trial data to CSV
def save_trial_data(data_file, header, row_dict):
    safe_row = {key: str(row_dict.get(key, "NA")) for key in header}
    write_header = not os.path.exists(data_file)
    with open(data_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)
        
# Collect demographic info
participant_dialog = gui.Dlg(title="Participant Info")
participant_dialog.addField("Gender", choices=["Male", "Female", "Other"])
participant_dialog.addField("Age")
participant_dialog.addField("Handedness", choices=["Left", "Right", "Ambidextrous"])
participant_dialog.show()
if not participant_dialog.OK:
    core.quit()
gender = participant_dialog.data[0]
age_raw = participant_dialog.data[1]
handedness = participant_dialog.data[2]
age = int(age_raw) if age_raw.isdigit() else "NA"


# Set up output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_file = os.path.join(save_directory, f"{participant_id_clean}_{timestamp}_experiment_data.csv")

# Fixation cross
fixation_cross = visual.TextStim(win, text="+", color='white', height=30)

# Response key mappings
left_vividness_keys = {'a': 1, 'z': 2, 'e': 3, 'r': 4} # counterbalancing: the keys switch -- half the time vividness is on the left hand
right_vividness_keys = {'u': 1, 'i': 2, 'o': 3, 'p': 4}
left_confidence_keys = {'a': 1, 'z': 2, 'e': 3, 'r': 4}
right_confidence_keys = {'u': 1, 'i': 2, 'o': 3, 'p': 4}
response_keys = ['v', 'b', 'escape']

# Create visual scales for both conditions
def draw_visual_scale(win, selected, labels, y_offset=-100):
    spacing = 150
    start_x = -((len(labels) - 1) * spacing) / 2
    rect_size = 50

    for i, label in enumerate(labels):
        x = start_x + i * spacing
        # Draw square
        color = 'blue' if selected == (i + 1) else None
        square = visual.Rect(win, width=rect_size, height=rect_size,
                             lineColor='white', fillColor=color, pos=(x, y_offset))
        square.draw()

        # Draw label below the square
        text = visual.TextStim(win, text=label, pos=(x, y_offset - 40),
                               height=16, color='white', wrapWidth=200)
        text.draw()

# Arrow stimuli for mental replay vs non-replay
def make_arrow(win, pos, angle): # create an arrow stimulus at position 'pos' rotated by 'angle' degrees
    return visual.ShapeStim(
        win,
        vertices=[[-40, -5], [0, -5], [0, -20], [35, 0], [0, 20], [0, 5], [-40, 5]],
        fillColor='black',
        lineColor='black',
        pos=pos,
        ori=angle
    )

# Positions (corners)
arrow_positions = [(-100, 100), (100, 100), (-100, -100), (100, -100)]

# Angles
angles_inward = [45, 135, -45, -135]    # for mental replay
angles_outward = [-135, -45, 135, 45]  # for non-replay

# Generate arrow stimuli lists
arrows_inward = [make_arrow(win, pos, angle) for pos, angle in zip(arrow_positions, angles_inward)]
arrows_outward = [make_arrow(win, pos, angle) for pos, angle in zip(arrow_positions, angles_outward)]


# CSV header
header = [
    "participant_id", "gender", "age", "handedness",
    "block", "block_type", "block_number", "trial", "global_trial",
    "response", "reference", "vividness", "confidence", "response_time",
    "gabor_direction", "correct", "stim_strength", "vividness_on_left"
]

# Main trial procedure
def run_trial(block_type, block_number, trial_num, global_trial, gabor_direction, stim_strength, stim_duration,
              give_feedback=False, saved_block_label=None):
    if saved_block_label is None:
        saved_block_label = block_type
    response = None

    # Set rating key mappings
    vividness_on_left = not left_for_confidence
    if left_for_confidence:
        confidence_keys = left_confidence_keys
        vividness_keys = right_vividness_keys
        confidence_prompt_text = "How confident are you?\n\nUse A/Z/E/R (left hand)"
        vividness_prompt_text = "How vivid was your mental replay?\n\nUse U/I/O/P (right hand)"
    else:
        confidence_keys = right_confidence_keys
        vividness_keys = left_vividness_keys
        confidence_prompt_text = "How confident are you?\n\nUse U/I/O/P (right hand)"
        vividness_prompt_text = "How vivid was your mental replay?\n\nUse A/Z/E/R (left hand)"


    # Store prompts in proper order
    prompts = [("confidence", confidence_prompt_text, confidence_keys),
                ("vividness", vividness_prompt_text, vividness_keys)]
    
    # Have the training progressively by including more prompts
    parts = saved_block_label.split("_")
    if "training" in saved_block_label and len(parts) > 1 and parts[1].isdigit():
        block_num = int(parts[1])
        if block_num == 1:
            prompts = []
        elif block_num == 2:
            prompts = [("confidence", confidence_prompt_text, confidence_keys)]
        elif block_num == 3:
            prompts = [("confidence", confidence_prompt_text, confidence_keys),
                       ("vividness", vividness_prompt_text, vividness_keys)]

    # Fixation cross
    fixation_cross.draw()
    win.flip()
    check_for_escape()
    core.wait(FIXATION_CROSS_DURATION)

    # Stimulus presentation
    reference = np.random.randint(0, 180)
    to_show = gabor.generate_gabor_patches(win, reference=reference,
                                           direction=gabor_direction, distance_to_bound=stim_strength)
    correct_response = "v" if gabor_direction < 0 else "b"
    
    # Show the gabor patches as well as the arc for the answer
    for stim in to_show[0]:
        stim.draw()
    for stim in to_show[1]:
        stim.draw()
    win.flip()

    # utils.take_picture(win, "stim")
    check_for_escape()
    core.wait(stim_duration)
    
    event.clearEvents()

    # Just show the arc for the answer
    for stim in to_show[1]:
        stim.draw()
    win.flip()

    rt_clock.reset()

    while True:  # Wait for participant's response
        check_for_escape()
        key = event.getKeys(keyList=key_list, timeStamped=rt_clock)
        if key != []:
            info_response = key[0]
            response = info_response[0]
            response_time = info_response[1]
            break

    correct = correct_response == response

    # Mental replay pause using arrows
    if block_type == "with_mental_replay":
        for arrow in arrows_inward:
            arrow.draw()
        win.flip()
        check_for_escape()
        core.wait(MENTAL_REPLAY_PAUSE)

    elif block_type == "without_mental_replay":
        for arrow in arrows_outward:
            arrow.draw()
        win.flip()
        check_for_escape()
        core.wait(MENTAL_REPLAY_PAUSE)

    # Ratings loop
    ratings = {"vividness": "NA", "confidence": "NA"}
    for measure, prompt_text, keymap in prompts:
        prompt = visual.TextStim(win, text=prompt_text, color='white', height=20, wrapWidth=700)
        rating = None
        event.clearEvents()
        labels = ["Not vivid", "Slightly", "Moderately", "Very vivid"] if measure == "vividness" \
                    else ["Not confident", "Slightly", "Moderately", "Very confident"]
        while rating is None:
            check_for_escape()
            keys = event.getKeys()
            for key in keys:
                if key in keymap:
                    rating = keymap[key]
            # Draw prompt and highlight the current rating dynamically
            prompt.draw()
            utils.draw_visual_scale(win, selected=rating, labels=labels, y_offset=-200) # pass current rating to highlight
            win.flip()
        core.wait(0.5)

        ratings[measure] = rating

    # Feedback
    if give_feedback:
        feedback_text = 'Correct!' if correct else 'Wrong!' if response else 'No response'
        feedback = visual.TextStim(win, text=feedback_text, pos=(0, 0), color='white')
        feedback.draw()
        win.flip()
        check_for_escape()
        core.wait(FEEDBACK_TIME)

    # Save data for this trial
    row_dict = {
        "participant_id": participant_id,
        "gender": gender,
        "age": age,
        "handedness": handedness,
        "block": saved_block_label,
        "block_type": block_type,
        "block_number": block_number,
        "trial": trial_num,
        "global_trial": global_trial,
        "response": response,
        "reference": reference,
        "vividness": ratings["vividness"],
        "confidence": ratings["confidence"],
        "response_time": response_time,
        "gabor_direction": gabor_direction,
        "correct": correct,
        "stim_strength": stim_strength,
        "vividness_on_left": vividness_on_left if block_type == "with_mental_replay" else None
    }
    save_trial_data(data_file, header, row_dict)

    return correct

# TRAINING PHASE (Adaptive & Varying Duration)
def training_phase():
    utils.show_images(win, image_intro, min_display_time)

    utils.show_images(win, image_training1, min_display_time)
    training1_directions = []
    baseline_trials = 10 # Not 20 because it is quite easy now
    max_extra_trials = 10
    accuracy_threshold = 0.8
    trial_counter = 0
    correct_history = []

    # Long presentation time
    while True:
        direction = utils.get_pseudorandom_direction(prev_directions=training1_directions)
        training1_directions.append(direction)

        correct = run_trial(
            block_type="training",
            block_number=0,
            trial_num=trial_counter + 1,
            global_trial=trial_counter + 1,
            gabor_direction=direction,
            stim_strength=max(20-trial_counter, 10),
            give_feedback=True,
            saved_block_label="training_1",
            stim_duration = 1.0,
        )
        correct_history.append(correct)
        trial_counter += 1

        if trial_counter >= baseline_trials:
            recent_accuracy = sum(correct_history[-baseline_trials:]) / baseline_trials
            if recent_accuracy >= accuracy_threshold or trial_counter >= baseline_trials + max_extra_trials:
                break

    # 2: non-mental replay practice
    utils.show_images(win, image_training2, min_display_time)
    training2_directions = []
    trial_counter = 0
    correct_history = []

    while True:
        direction = utils.get_pseudorandom_direction(prev_directions=training2_directions)
        training2_directions.append(direction)

        correct = run_trial(
            block_type="without_mental_replay",
            block_number=0,
            trial_num=trial_counter + 1,
            global_trial=trial_counter + 1 + len(training1_directions),
            gabor_direction=direction,
            stim_strength=max(20-trial_counter, 10),
            give_feedback=True,
            saved_block_label="training_2",
            stim_duration = 0.7
        )
        correct_history.append(correct)
        trial_counter += 1

        if trial_counter >= baseline_trials:
            recent_accuracy = sum(correct_history[-baseline_trials:]) / baseline_trials
            if recent_accuracy >= accuracy_threshold or trial_counter >= baseline_trials + max_extra_trials:
                break

    # 3: mental replay practice
    utils.show_images(win, image_training3, min_display_time)
    training3_directions = []
    trial_counter = 0
    correct_history = []

    while True:
        direction = utils.get_pseudorandom_direction(prev_directions=training3_directions)
        training3_directions.append(direction)

        correct = run_trial(
            block_type="with_mental_replay",
            block_number=0,
            trial_num=trial_counter + 1,
            global_trial=trial_counter + 1 + len(training1_directions) + len(training2_directions),
            gabor_direction=direction,
            stim_strength=max(20-trial_counter, 10),
            give_feedback=True,
            saved_block_label="training_3",
            stim_duration = 0.4
        )
        correct_history.append(correct)
        trial_counter += 1

        if trial_counter >= baseline_trials:
            recent_accuracy = sum(correct_history[-baseline_trials:]) / baseline_trials
            if recent_accuracy >= accuracy_threshold or trial_counter >= baseline_trials + max_extra_trials:
                break

    utils.show_images(win, image_training_end, min_display_time)


# EXPERIMENTAL PHASE (Adaptive staircase)
def exp_phase():
    # Loop through blocks
    stim_strength = 10  # initial coherence/distance value
    for block_idx, condition in enumerate(block_order):

        # Break halfway
        if block_idx == TOTAL_BLOCKS // 2:
            break_clock = core.Clock()
            
            # Show break image
            utils.show_images(win, image_break, min_display_time)
            event.clearEvents()
            event.waitKeys()
            break_duration = break_clock.getTime()
            
            # Save break info in CSV
            row_dict = {
                "participant_id": participant_id,
                "gender": gender,
                "age": age,
                "handedness": handedness,
                "block": "break",
                "block_type": "pause",
                "block_number": block_idx,
                "trial": "NA",
                "global_trial": "NA",
                "response": "NA",
                "reference": "NA",
                "vividness": "NA",
                "confidence": "NA",
                "response_time": break_duration,  # <-- actual time spent
                "gabor_direction": "NA",
                "correct": "NA",
                "stim_strength": "NA",
                "vividness_on_left": "NA"
            }
            save_trial_data(data_file, header, row_dict)
            
        # Run all trials in block
        for trial in range(TRIALS_PER_BLOCK):
            win.flip()
            core.wait(INTERTRIAL_PAUSE)

            # Get next direction (pseudo-randomized to avoid streaks)
            direction = utils.get_pseudorandom_direction(prev_directions=test_directions)
            print(direction)
            test_directions.append(direction)

            block_number = block_idx + 1
            global_trial_number = block_idx * TRIALS_PER_BLOCK + trial + 1

            run_trial(
                block_type=condition,
                block_number=block_number,
                trial_num=trial + 1,
                global_trial=global_trial_number,
                gabor_direction=direction,
                stim_strength=stim_strength,
                stim_duration=STIMULUS_DURATION,
            )

#training_phase()
#utils.show_images(win, image_exp_instructions, min_display_time) 
exp_phase()
utils.show_images(win, image_end, min_display_time)
win.close()
core.quit()
