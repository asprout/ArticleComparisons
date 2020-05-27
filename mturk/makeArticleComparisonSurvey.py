import numpy as np
import pandas as pd
import random

# Import individual XML item classes
from Question import Question, Category, Options
from XMLdoc import XMLdoc, Module, Task, Document, HIT, Set
from createSurvey import Survey 

consent_form = '''
<h1>Microsoft Research Project Participation Consent Form</h1>
<h2>INTRODUCTION</h2>
Thank you for taking the time to consider volunteering in a Microsoft Corporation research project.  This form explains what would happen if you join this research project. 
Please read it carefully and take as much time as you need. Email the study team to ask about anything that is not clear.  
Participation in this study is voluntary and you may withdraw at any time. 
  
<h2>TITLE OF RESEARCH PROJECT</h2>
Article Labeling Task
  
<h3>Principal Investigator</h3>
David Rothschild
  
<h2>PURPOSE</h2>
The purpose of this project is to collect data on the content of URLs as they pertain to various topics.

<h2>PROCEDURES</h2>
<p>During this session, the following will happen: </p>
<p>For each task, you will be given a URL to visit. Your job is to visit the URL and answer the given questions about the URL content.</p>
<p>While each task is expected to take under a minute on average, the time you spend will vary depending on the length of each individual article.</p>

<h3>PAYMENT FOR PARTICIPATION</h3>
<p>You will receive $0.1 for completing this session, and additional bonus payments ($0.02 for each question) contingent on your performance, as measured by the agreement of your responses with those of other raters.</p>

<h2>PERSONAL INFORMATION</h2>
<p>Aside from your WorkerID, we do not collect any personal information in this project. </p>
<p>Your WorkerID and response will be temporarily recorded and used for the purpose of paying out bonuses based on task performance.</p>
<p>Your WorkerID will not be shared outside of Microsoft Research and the confines of this study without your permission, and will be promptly deleted after compensation has been successfully provided (30 days or less). De-identified data may be used for future research or given to another investigator for future use without additional consent. </p>
<p>Responses from all participants will be aggregated and stored for a period of up to 5 years. Once your WorkerID is disassociated from your responses we may not be to remove your data from the study without re-identifying you.</p>
<p>For additional information on how Microsoft handles your personal information, 
please see the <a href="https://privacy.microsoft.com/en-us/privacystatement">Microsoft Privacy Statement</a>.</p>

<h2>BENEFITS AND RISKS</h2>
<p>Benefits: The research team expects to learn about the representation of various topics in online news. You will receive any public benefit that may come these Research Results being shared with the greater scientific community. </p>
<p>Risks:  During your participation, you should experience no greater risks than in normal daily life. 
However, there is a small possibility that you may be exposed to sensitive and violent content through the article content. <p/>
<p>You accept the risks described above and whatever consequences may come of those risks, however unlikely, 
unless caused by our negligence or intentional misconduct.  
You hereby release Microsoft and its affiliates from any claim you may have now or in the future arising from such risks or consequences.    
In addition, you agree that Microsoft will not be liable for any loss, damages or injuries 
that may come of improper use of the study prototype, equipment, facilities, or 
any other deviations from the instructions provided by the research team.   
Don’t participate in this study if you feel you may not be able to safely participate in any way 
including due to any physical or mental illness, condition or limitation.    
You agree to immediately notify the research team of any incident or issue or unanticipated risk or incident.</p>

<h2>CONTACT INFORMATION</h3>
<p>Should you have any questions concerning this project, or if you are harmed as a result of being in this study, please contact David Rothschild at davidmr@microsoft.com.</p>
<p>Should you have any questions about your rights as a research subject, please contact Microsoft Research Ethics Program Feedback at MSRStudyfeedback@microsoft.com.</p>
<p>Upon request, a copy of this consent form will be provided to you for your records. On behalf of Microsoft, we thank you for your contribution and look forward to your research session.</p>
'''

if __name__ == '__main__':
	header = '''
		<head>
		<style type = "text/css">
			.doc {
				margin: 2em;
			}
		</style>
		</head>
	'''

	instructions = '''
		<p> 
		Instructions 
		</p>
	'''

	ntasks = 100
	assignments_per_task = 3
	tasks_per_assignment = 10 
	print(f"Creating mturk survey with {ntasks} tasks, {assignments_per_task} assignments per task, {tasks_per_assignment} tasks per assignment")

	### Questions 
	q_consent = Question(varname = "consent", 
		questiontext = "Please carefully read the consent form provided before proceeding with this task.", 
		valuetype = "categorical",
		categories = ["I agree to participate in this study.", "I do not agree to participate in this study."], 
		values = [1, 0])

	q_instructions = Question(varname = "instructions", 
					questiontext = "Please take time to read the instructions carefully before you proceed.",
					valuetype = "categorical", 
					categories = ["I have read and understood the task instructions."])

	q_label = Question(varname = "label",
					questiontext = "This article is very likely to appeal to someone interested in the flu vaccine.",
					options = Options(layout = "horizontal", 
						lowLabel = "Strongly Disagree", highLabel = "Strongly Agree", 
						outsideCategories = 'N/A (invalid article)'), 
					valuetype = "categorical",
					categories = [-2, -1, 0, 1, 2], 
					helptext = "Pick the choice to the best of your judgment. You will be paid based on the agreement of your labels with those of other raters!",
					bonus = ["threshold:50", 1])

	### Modules 
	module_consent = Module(name = "consent", header = "Consent", questions = [q_consent])
	module_instructions = Module(name = "instructions", header = "Instructions", questions = [q_instructions])
	module_label = Module(name = "label", header = "Task", questions = [q_label])

	### TASKS 
	task_consent = Task(name = "consent", taskid = 0, modules = module_consent, 
		content = header + '<div class="doc">' + consent_form + "</div>")

	task_instructions = Task(name = "instructions", taskid = 1, modules = module_instructions, 
		content = header + '<div class="doc">' + instructions + "</div>")

	tasklist_label = [Task(name = f"label_task{i}", taskid = i, modules = module_label, 
		content = header + '<div class="doc">' + "[Article Content Here]" + "</div>") for i in range(10, ntasks + 10)]

	XMLdoc = XMLdoc(modules = [module_consent, module_instructions, module_label], 
		tasks = [task_consent, task_instructions] + tasklist_label, 
		hits = Survey(ntasks, assignments_per_task, tasks_per_assignment).get_hit_list(
			start_task_index = 10, 
			perm_tasks = [0, 1]))
	XMLdoc.add_documents([t.document for t in XMLdoc.tasks])

	for hit in XMLdoc.hits:
		hit.add_taskcondition(1, "0*consent*consent>=1")
		for taskid in hit.task_list(start_task_index = 10):
			hit.add_taskcondition(taskid, "0*consent*consent>=1")

	print(f"{len(XMLdoc.hits)} total hits created")
	XMLdoc.write_xml("flu_survey.xml")