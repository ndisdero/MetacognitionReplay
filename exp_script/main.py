from psychopy import visual, core, event, gui, prefs
import random
import os
from datetime import datetime
import utils
import gabor_patches.generate_gaborPatches as gabor
import numpy as np


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
N_TRAINING_TRIALS = 4  # number of training trials
TOTAL_BLOCKS = 4  # must be even: half with replay, half without
NUMBER_OF_TRIALS = 2  # per block
STIMULUS_DURATION = 0.4
INTERTRIAL_PAUSE = 2  # fixation cross display time
MENTAL_REPLAY_PAUSE = 3  # pause during mental replay instruction
FEEDBACK_TIME = 1
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

image_task1 = [os.path.join(image_dir, f) for f in ["Task1_1.JPG"]] # mental replay screen
image_task2 = [os.path.join(image_dir, f) for f in ["Task2_1.JPG"]] # non-mental replay screen
image_end = [os.path.join(image_dir, f) for f in ["End.JPG"]]
image_replay = [os.path.join(image_dir, "MentalReplayImg.png")]

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
participant_id_num = int(participant_id_clean)  # participant ID must be numeric
group_id = participant_id_num % 8  # assign participant to 1 of 8 groups (0–7)

# Factor 1: starting condition
start_with_replay = (group_id % 2 == 0)  # even groups: replay first, odd groups: non-replay first

# Factor 2: scale order
vividness_first = (group_id // 2) % 2 == 0  # groups 0–1,4–5 vividness first; groups 2–3,6–7 confidence first

# Factor 3: hand assignment
left_for_confidence = (group_id // 4) % 2 == 0  # groups 0–3: left hand = confidence; groups 4–7: left hand = vividness

half_blocks = TOTAL_BLOCKS // 2
if start_with_replay:
    block_order = ["with_mental_replay"] * half_blocks + ["without_mental_replay"] * half_blocks
else:
    block_order = ["without_mental_replay"] * half_blocks + ["with_mental_replay"] * half_blocks

test_directions = []

print(f"Assigned to group {group_id}: "
      f"{'Replay first' if start_with_replay else 'Non-replay first'}, "
      f"{'Vividness first' if vividness_first else 'Confidence first'}, "
      f"{'Left=confidence' if left_for_confidence else 'Left=vividness'}")


# Step 2: Collect demographic info
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
dot_instruction = visual.TextStim(
    win, text="Which direction are most of the dots moving?\nPress V or B.",
    color='white', height=20, pos=(0, -250), wrapWidth=700
)
replay_text = visual.TextStim(win, text="Mentally replay the motion you just saw.", color='white', height=20)

# Response key mappings
left_vividness_keys = {'a': 1, 'z': 2, 'e': 3, 'r': 4} # counterbalancing: the keys switch -- half the time vividness is on the left hand
right_vividness_keys = {'u': 1, 'i': 2, 'o': 3, 'p': 4}
left_confidence_keys = {'a': 1, 'z': 2, 'e': 3, 'r': 4}
right_confidence_keys = {'u': 1, 'i': 2, 'o': 3, 'p': 4}
response_keys = ['v', 'b', 'escape']

# CSV header
header = [
    "participant_id", "gender", "age", "handedness",
    "block", "block_type", "block_number", "trial", "global_trial",
    "response", "reference", "vividness", "confidence", "response_time",
    "gabor_direction", "correct", "stim_strength", "vividness_on_left"
]

# Main trial procedure
def run_trial(block_type, block_number, trial_num, global_trial, gabor_direction, stim_strength,
              give_feedback=False, saved_block_label=None):
    if saved_block_label is None:
        saved_block_label = block_type   
    response = None
    
    # Set rating key mappings
    if block_type == "with_mental_replay":
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
    else:  # without mental replay
        vividness_on_left = None
        confidence_keys = right_confidence_keys
        confidence_prompt_text = (
            "How confident are you in your response?\n\n"
            "U = Not confident   I = Slightly   O = Moderately   P = Very confident"
        )
        vividness_keys = None
        vividness_prompt_text = None

    # Store prompts in proper order
    prompts = []
    if block_type == "with_mental_replay":
        if vividness_first:
            prompts = [("vividness", vividness_prompt_text, vividness_keys),
                       ("confidence", confidence_prompt_text, confidence_keys)]
        else:
            prompts = [("confidence", confidence_prompt_text, confidence_keys),
                       ("vividness", vividness_prompt_text, vividness_keys)]
    elif block_type == "without_mental_replay":
        prompts = [("confidence", confidence_prompt_text, confidence_keys)]

    # Fixation cross
    fixation_cross.draw()
    win.flip()
    check_for_escape()
    core.wait(FIXATION_CROSS_DURATION)

    # Stimulus presentation
    reference = np.random.randint(0, 180)
    to_show = gabor.generate_gabor_patches(win, reference=reference,
                                           direction=gabor_direction, distance_to_bound=stim_strength)
    
    for stim in to_show[0]:
        stim.draw()
    for stim in to_show[1]:
        stim.draw()
    win.flip()
    check_for_escape()
    core.wait(STIMULUS_DURATION)

    response_time = rt_clock.getTime()
    correct_response = 'v' if gabor_direction < 0 else 'b'

    event.clearEvents()

    for stim in to_show[1]:
        stim.draw()
    win.flip()
    check_for_escape()

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

    # Ratings loop
    ratings = {"vividness": "NA", "confidence": "NA"}

    if block_type == "with_mental_replay":
        # Show the replay instruction IMAGE instead of text
        utils.show_images(win, image_replay, min_display_time)
        core.wait(MENTAL_REPLAY_PAUSE)
    
    for measure, prompt_text, keymap in prompts:
        prompt = visual.TextStim(win, text=prompt_text, color='white', height=20, wrapWidth=700)
        rating = None
        event.clearEvents()
        while rating is None:
            check_for_escape()
            prompt.draw()
            labels = ["Not vivid", "Slightly", "Moderately", "Very vivid"] if measure == "vividness" \
                     else ["Not confident", "Slightly", "Moderately", "Very confident"]
            utils.draw_visual_scale(win, None, labels)
            win.flip()
            check_for_escape()
            keys = event.getKeys()
            for key in keys:
                if key in keymap:
                    rating = keymap[key]
        ratings[measure] = rating

    # Feedback
    if give_feedback:
        feedback_text = 'Correct!' if correct else 'Wrong!' if response else 'No response'
        feedback = visual.TextStim(win, text=feedback_text, pos=(0, 0), color='white')
        feedback.draw()
        win.flip()
        check_for_escape()
        core.wait(FEEDBACK_TIME)

    # Save data
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
    utils.save_trial_data(data_file, header, row_dict)

    return correct


# TRAINING PHASE (Adaptive & Varying Duration)
def training_phase():
    utils.show_images(win, image_intro, min_display_time)

    # 1: basic dot motion discrimination (left/right only)
    utils.show_images(win, image_training1, min_display_time)
    training1_directions = []
    baseline_trials = 20
    max_extra_trials = 10
    accuracy_threshold = 0.85
    trial_counter = 0
    correct_history = []

    while True:
        direction = utils.get_pseudorandom_direction(prev_directions=training1_directions)
        training1_directions.append(direction)
        global STIMULUS_DURATION
        STIMULUS_DURATION = 1.0

        correct = run_trial(
            block_type="training",
            block_number=0,
            trial_num=trial_counter + 1,
            global_trial=trial_counter + 1,
            gabor_direction=direction,
            stim_strength=20,
            give_feedback=True,
            saved_block_label="training_1"
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
        STIMULUS_DURATION = 1.0

        correct = run_trial(
            block_type="without_mental_replay",
            block_number=0,
            trial_num=trial_counter + 1,
            global_trial=trial_counter + 1 + len(training1_directions),
            gabor_direction=direction,
            stim_strength=20,
            give_feedback=True,
            saved_block_label="training_2"
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
        STIMULUS_DURATION = 0.5

        correct = run_trial(
            block_type="with_mental_replay",
            block_number=0,
            trial_num=trial_counter + 1,
            global_trial=trial_counter + 1 + len(training1_directions) + len(training2_directions),
            gabor_direction=direction,
            stim_strength=20,
            give_feedback=True,
            saved_block_label="training_3"
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
    for block_idx, condition in enumerate(block_order):
        if condition == "without_mental_replay":
            utils.show_images(win, image_task1, min_display_time)
        else:
            utils.show_images(win, image_task2, min_display_time)

        stim_strength = 20
        for trial in range(NUMBER_OF_TRIALS):
            win.flip()
            core.wait(INTERTRIAL_PAUSE)

            direction = utils.get_pseudorandom_direction(prev_directions=test_directions)
            test_directions.append(direction)

            block_number = block_idx + 1
            global_trial_number = block_idx * NUMBER_OF_TRIALS + trial + 1

            run_trial(
                block_type=condition,
                block_number=block_number,
                trial_num=trial + 1,
                global_trial=global_trial_number,
                gabor_direction=direction,
                stim_strength=stim_strength,
            )

            # Adapt stim_strength
            try:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                    last_row = lines[-1].strip().split(',')
                    last_correct_str = last_row[14]  # column index 14 = correct
                    last_correct = True if last_correct_str == 'True' else False
            except Exception:
                last_correct = None

            step_up = 1.0   # after incorrect
            step_down = 0.5 # after correct

            if last_correct is True:
                stim_strength = max(0.1, stim_strength - step_down)
            elif last_correct is False:
                stim_strength = min(100, stim_strength + step_up)


training_phase()
exp_phase()
utils.show_images(win, image_end, min_display_time)
win.close()
core.quit()
