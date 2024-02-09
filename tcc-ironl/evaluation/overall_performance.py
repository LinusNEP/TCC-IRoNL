import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score

# Interaction data
llm_data_log_path = 'llm_data_log.csv'
nsr_log_path = 'nsr_log.csv'
oia_log_path = 'oia_log.csv'

llm_data_log = pd.read_csv(llm_data_log_path)
nsr_log = pd.read_csv(nsr_log_path)
oia_log = pd.read_csv(oia_log_path)
llm_data_log_head = llm_data_log.head()
nsr_log_head = nsr_log.head()
oia_log_head = oia_log.head()

llm_data_log_head, nsr_log_head, oia_log_head

# Group of synonymous labels
synonym_groups = {
    'stop_group': ['stop', 'hold', 'pause', 'wait','UNKNOWN','unknown', 'halt'],
    'backward_group': ['back', 'backward'],
    'forward_group': ['front', 'forward'],
    'left_group': ['left', 'turn left', 'turn_left'],
    'right_group': ['right', 'turn right', 'turn_right'],
    'professor_group': ['elmar', 'professor', 'rueckert'],
    'secretary_group': ['regina', 'Regina', 'Secretary', 'secretary'],
    'office_group': ['linus', 'vedant', 'niko', 'fotios', 'melanie', 'conference'],
    'technician_group': ['konrad', 'technician', 'server', 'printer' , 'workshop'],
    'kitchen_group': ['kitchen'],
    'circle_group': ['circle', 'spiral', 'circular', 'clockwise'],
    'elevator_group': ['lift', 'elevator', 'toilet', 'passage']
}

def map_label_to_group(label):
    for group_name, synonyms in synonym_groups.items():
        if label.lower() in synonyms:
            return group_name
    return 'other_group'  

llm_data_log['Predicted Label Group'] = llm_data_log['Predicted Label'].apply(map_label_to_group)
llm_data_log['True Label Group'] = llm_data_log['True Label'].apply(map_label_to_group)

llm_accuracy = (llm_data_log['Predicted Label Group'] == llm_data_log['True Label Group']).mean()
nsr_success_rate = nsr_log['Success'].mean()
oia_precision = oia_log['Correct Identification'].mean()

data_for_plot = {
    'CRA': [llm_accuracy],
    'NSR': [nsr_success_rate],
    'OIA': [oia_precision]
}

performance_df = pd.DataFrame(data_for_plot)

nsr_log['Success'] = nsr_log['Success'].astype(float)
oia_log['Correct Identification'] = oia_log['Correct Identification'].astype(float)

nsr_std = nsr_log['Success'].std()
oia_std = oia_log['Correct Identification'].std()

llm_label_variance = llm_data_log['Predicted Label'].value_counts(normalize=True).var()
"""
# For NSR Log, calculate the variance in the frequency of successes
nsr_success_frequency = nsr_log['Success'].value_counts(normalize=True)
nsr_std = nsr_success_frequency.var()
# For OIA Log, calculate the variance in the frequency of correct identifications
oia_correct_freq = oia_log['Correct Identification'].value_counts(normalize=True)
oia_std = oia_correct_freq.var()
"""

std_for_plot = {
    'CRA': [llm_label_variance],
    'NSR': [nsr_std],
    'OIA': [oia_std]
}
std_df = pd.DataFrame(std_for_plot, index=[0])

true_labels = llm_data_log['True Label']
predicted_labels = llm_data_log['Predicted Label']
cm = confusion_matrix(true_labels, predicted_labels, labels=true_labels.unique())

def consolidate_labels(label):
    label = label.lower()
    if "professor" in label or "elmar" in label:
        return "Professor"
    elif "secretary" in label or "regina" in label or "Regina" in label or "Secretary" in label:
        return "Secretary"
    elif "linus" in label or "vedant" in label or "niko" in label or "fotios" in label or "conference" in label or "student" in label:
        return "Offices"
    elif "back" in label or "backward" in label:
        return "Backward"
    elif "forward" in label or "front" in label:
        return "Forward"
    elif "right" in label or "turn_right" in label or "turn right" in label:
        return "Right"
    elif "left" in label or "turn_left" in label or "turn left" in label:
        return "Left"
    elif "kitchen" in label:
        return "Kitchen"
    elif "circle" in label or "circular" in label or "spiral" in label or "clockwise" in label or "c_spin" in label:
        return "Circle"
    elif "lift" in label or "elevator" in label or "toilet" in label or "passage" in label:
        return "Elevator"
    elif "konrad" in label or "server" in label or "workshop" in label or "printer" in label or "technician" in label:
        return "Workshop"
    elif "unknown" in label or "hold" in label or "pause" in label or "wait" in label or "halt" in label or "stop" in label or "(none, none)" in label:
        return "Stop"
    else:
        return label

llm_data_log['Consolidated True Label'] = llm_data_log['True Label'].apply(consolidate_labels)
llm_data_log['Consolidated Predicted Label'] = llm_data_log['Predicted Label'].apply(consolidate_labels)

consolidated_true_labels = llm_data_log['Consolidated True Label']
consolidated_predicted_labels = llm_data_log['Consolidated Predicted Label']
cm_consolidated = confusion_matrix(consolidated_true_labels, consolidated_predicted_labels, labels=consolidated_true_labels.unique())

# Overall accuracy
overall_accuracy = accuracy_score(consolidated_true_labels, consolidated_predicted_labels)
unique_labels = consolidated_true_labels.unique()
label_accuracies = {}
for label in unique_labels:
    label_data = llm_data_log[llm_data_log['Consolidated True Label'] == label]
    label_accuracy = accuracy_score(label_data['Consolidated True Label'], label_data['Consolidated Predicted Label'])
    label_accuracies[label] = label_accuracy
mean_accuracy = sum(label_accuracies.values()) / len(label_accuracies)
overall_accuracy, label_accuracies, mean_accuracy
print(mean_accuracy)

# Questionnaire data
file_path = 'questionnaire_responses.csv'
questionnaire_data = pd.read_csv(file_path)
questionnaire_data.head()

questionnaire_data['Gender'] = questionnaire_data['Gender'].str.lower().map({'male': 'Male', 'm': 'Male', 'männlich': 'Male', 'female': 'Female', 'f': 'Female', 'weiblich': 'Female'}).fillna('Other')
occupation_normalization = {'student': 'Students', 'Student': 'Students', 'Bachelor Student': 'Students', 'University Assistant': 'Students', 'Master degree student': 'Students', 'researcher': 'Researcher' , 'Researcher': 'Researcher'}
questionnaire_data['Occupation'] = questionnaire_data['Occupation'].replace(occupation_normalization)

#sns.set(style="whitegrid")
#plt.rcParams.update({'font.size': 20})

respondent_count = questionnaire_data.groupby('Occupation').size()

questionnaire_data['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'] = pd.to_numeric(questionnaire_data['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'], errors='coerce')
questionnaire_data['How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'] = pd.to_numeric(questionnaire_data['How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'], errors='coerce')
questionnaire_data['How intuitive did you find the process of giving commands to the robot? (1 - Not intuitive, 5 - Highly intuitive)'] = pd.to_numeric(questionnaire_data['How intuitive did you find the process of giving commands to the robot? (1 - Not intuitive, 5 - Highly intuitive)'], errors='coerce')

response_mapping = {
    'Never': 1,
    'Rarely': 2,
    'Sometimes': 3,
    'Most of the time': 4,
    'Always': 5
}
questionnaire_data['Did you feel that the robot understood your commands accurately?'] = pd.to_numeric(questionnaire_data['Did you feel that the robot understood your commands accurately?'].map(response_mapping), errors='coerce')

questionnaire_data['How satisfied are you with the responsiveness of the robot to your commands? (1 - Very dissatisfied, 5 - Very satisfied)'] = pd.to_numeric(questionnaire_data['How satisfied are you with the responsiveness of the robot to your commands? (1 - Very dissatisfied, 5 - Very satisfied)'], errors='coerce')

grouped_data = questionnaire_data.groupby('Occupation').agg({
    'Familiarity with Technology (1 - Not familiar, 5 - Very familiar)': 'mean',
    'How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)': 'mean',
    'Did you feel that the robot understood your commands accurately?': 'mean',
    'How intuitive did you find the process of giving commands to the robot? (1 - Not intuitive, 5 - Highly intuitive)': 'mean',
    'How satisfied are you with the responsiveness of the robot to your commands? (1 - Very dissatisfied, 5 - Very satisfied)': 'mean'
}).reset_index()

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[0.45, 1], figure=fig)
gs1 = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[0.70, 0.25], figure=fig)
ax0 = plt.subplot(gs[0, 0])  
ax1 = plt.subplot(gs[0, 1])  
ax2 = plt.subplot(gs1[1, 0])
ax3 = plt.subplot(gs1[1, 1])  

metric_names = list(data_for_plot.keys())
bar_width = 0.001
num_metrics = len(metric_names)
bar_positions = [i * (bar_width + 0.0001) for i in range(num_metrics)]
colors = ['#fed2a1', '#4c8986', '#F38080','#BDD7E0', '#C1B896', '#BDE0C6', '#E0BDD2', '#A4A4B3']

for i, metric in enumerate(metric_names):
    ax0.bar(bar_positions[i], performance_df.loc[0, metric], width=bar_width, color=colors[i % len(colors)],
            yerr=std_df.loc[0, metric], capsize=4)
            
ax0.set_ylabel('Performance (%)', fontsize=20, fontweight='bold')
ax0.set_xticks(bar_positions)
ax0.set_xticklabels(metric_names, rotation=45, fontsize=16, fontweight='bold')
ax0.tick_params(axis='y', labelsize=16, labelcolor='black')
ax0.grid(axis='y', linestyle='--', alpha=0.7)
ax0.set_xlabel('Metrics', fontsize=20, fontweight='bold')

for i, rect in enumerate(ax0.patches):
    height = rect.get_height()
    ax0.annotate(f'{height * 100:.2f}%', (rect.get_x() + rect.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=18, fontweight='bold')

cm_normalized = cm_consolidated.astype('float') / cm_consolidated.sum(axis=1)[:, np.newaxis]
ax_cm = ax1
sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap='Reds', ax=ax_cm,
            xticklabels=unique_labels, yticklabels=unique_labels, cbar=True, annot_kws={"size": 16})#, "weight": 'bold'})
plt.setp(ax_cm.get_xticklabels(), rotation=45, fontsize=16, fontweight='bold')
plt.setp(ax_cm.get_yticklabels(), rotation=0, fontsize=16, fontweight='bold')
ax1.set_xlabel('Predicted labels', fontsize=20, fontweight='bold')
ax1.set_ylabel('True labels', fontsize=20, fontweight='bold')
cbar = ax_cm.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)  
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold') 
    label.set_fontsize(16) 
plt.tight_layout()  

grouped_data['Total'] = grouped_data.sum(axis=1)
grouped_data_sorted = grouped_data.sort_values(by='Total', ascending=False)
index = np.arange(grouped_data_sorted.shape[0])
bar_width = 0.8
tech_bar = ax2.bar(index, grouped_data_sorted['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'], bar_width, color=colors[3], label='Familiarity with technology\n(1 - Not familiar, 5 - Very familiar)')
comm_bar = ax2.bar(index, grouped_data_sorted['How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'], bar_width, bottom=grouped_data_sorted['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'], color=colors[4], label='Ease of interacting with the robot\n(1 - Very difficult, 5 - Very easy)')
understand_bar = ax2.bar(index, grouped_data_sorted['Did you feel that the robot understood your commands accurately?'], bar_width, bottom=grouped_data_sorted['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'] + grouped_data_sorted['How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'], color=colors[5], label='Commands understanding accuracy\n(1 - Never, 5 - Always)')
intuitive_bar = ax2.bar(index, grouped_data_sorted['How intuitive did you find the process of giving commands to the robot? (1 - Not intuitive, 5 - Highly intuitive)'], bar_width, bottom=grouped_data_sorted['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'] + grouped_data_sorted['How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'] + grouped_data_sorted['Did you feel that the robot understood your commands accurately?'], color=colors[6],label='Intuitiveness of the framework\n(1 - Not intuitive, 5 - Highly intuitive)')
satisfaction_bar = ax2.bar(index, grouped_data_sorted['How satisfied are you with the responsiveness of the robot to your commands? (1 - Very dissatisfied, 5 - Very satisfied)'], bar_width, bottom=grouped_data_sorted['Familiarity with Technology (1 - Not familiar, 5 - Very familiar)'] + grouped_data_sorted['How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'] + grouped_data_sorted['Did you feel that the robot understood your commands accurately?'] + grouped_data_sorted['How intuitive did you find the process of giving commands to the robot? (1 - Not intuitive, 5 - Highly intuitive)'], color=colors[7],label='Response to commands satisfaction\n(1 - Very dissatisfied, 5 - Very satisfied)')

# Annotations
for idx, bars in enumerate(zip(tech_bar, comm_bar, understand_bar, intuitive_bar, satisfaction_bar)):
    tech, comm, understand, intuitive, satisfaction = bars
    total_height_tech = tech.get_height()
    total_height_comm = total_height_tech + comm.get_height()
    total_height_understand = total_height_comm + understand.get_height()
    total_height_intuitive = total_height_understand + intuitive.get_height()
    total_height_satisfaction = total_height_intuitive + satisfaction.get_height()
    
    # Annotation for each stacked bar segment
    ax2.text(tech.get_x() + tech.get_width() / 2, total_height_tech / 2, f'{grouped_data_sorted.iloc[idx, 1]:.2f}', ha='center', va='center', color='black', fontsize=20)
    ax2.text(comm.get_x() + comm.get_width() / 2, total_height_comm - comm.get_height() / 2, f'{grouped_data_sorted.iloc[idx, 2]:.2f}', ha='center', va='center', color='black', fontsize=20)
    ax2.text(understand.get_x() + understand.get_width() / 2, total_height_understand - understand.get_height() / 2, f'{grouped_data_sorted.iloc[idx, 3]:.2f}', ha='center', va='center', color='black', fontsize=20)
    ax2.text(intuitive.get_x() + intuitive.get_width() / 2, total_height_intuitive - intuitive.get_height() / 2, f'{grouped_data_sorted.iloc[idx, 4]:.2f}', ha='center', va='center', color='black', fontsize=20)
    ax2.text(satisfaction.get_x() + satisfaction.get_width() / 2, total_height_satisfaction - satisfaction.get_height() / 2, f'{grouped_data_sorted.iloc[idx, 5]:.2f}', ha='center', va='center', color='black', fontsize=20)
ax2.set_xlabel('Respondents occupation distribution', fontsize=20, fontweight='bold')
ax2.set_ylabel('Ratings', fontsize=20, fontweight='bold')
plt.subplots_adjust(bottom=0.14)
ax2.tick_params(axis='y', labelsize=16, labelcolor='black')
ax2.set_xticks(index)
ax2.set_xticklabels(grouped_data_sorted['Occupation'], rotation=30, fontsize=16, fontweight='bold')
ax2.legend()

handles, labels = ax2.get_legend_handles_labels()
legend = ax3.legend(handles, labels, loc='best', ncol=1, prop={'size': 18, 'weight': 'bold'})
ax3.axis('off')

ax2.get_legend().remove()
#plt.tight_layout()
plt.show()

# Percentage of participants who rated the ease of communication as 4 or 5 (favorable)
ease_communication_column = 'How would you rate the ease of communicating with the robot? (1 - Very difficult, 5 - Very easy)'
favorable_responses = questionnaire_data[ease_communication_column].apply(lambda x: x >= 4).sum()
total_responses = questionnaire_data[ease_communication_column].count()

percentage_favorable = (favorable_responses / total_responses) * 100
print(percentage_favorable)

# Percentage of participants who rated the intuitiveness of the approach as 4 or 5 (favorable)
intuitiveness_column = 'How intuitive did you find the process of giving commands to the robot? (1 - Not intuitive, 5 - Highly intuitive)'
intuitiveness_favorable_responses = questionnaire_data[intuitiveness_column].apply(lambda x: x >= 4).sum()
intuitiveness_total_responses = questionnaire_data[intuitiveness_column].count()

percentage_intuitiveness_favorable = (intuitiveness_favorable_responses / intuitiveness_total_responses) * 100
print(percentage_intuitiveness_favorable)

# Percentage of participants who are satisfied with the robot's response to the commands as 4 or 5 (favorable)
satisfaction_column = 'How satisfied are you with the responsiveness of the robot to your commands? (1 - Very dissatisfied, 5 - Very satisfied)'
satisfaction_favorable_responses = questionnaire_data[satisfaction_column].apply(lambda x: x >= 4).sum()
satisfaction_total_responses = questionnaire_data[satisfaction_column].count()

percentage_satisfaction_favorable = (satisfaction_favorable_responses / satisfaction_total_responses) * 100
print(percentage_satisfaction_favorable)


