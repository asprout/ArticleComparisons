import numpy as np
import pandas as pd
import random

# Import individual XML item classes
from Question import Question, Category, Options
from XMLdoc import XMLdoc, Module, Task, Document, HIT, Set

class Survey:
	""" 
	Evenly distributes nhits among nassignments such that:
	- each hit is assigned assignments_per_hit times
	- no assignment completes any two identical hits 
	"""
	def __init__(self, ntasks = 1000, assignments_per_task = 3,
							       tasks_per_assignment = 50):
		self.ntasks = ntasks # total number of tasks to be done
		self.assignments_per_task = assignments_per_task
		self.tasks_per_assignment = tasks_per_assignment

	def get_assignment_matrix(self, col_sum, row_sum, ncols):
		""" Given ncol tasks each to be done col_sum times, 
		and exactly row_sum unique tasks assigned per row, 
		returns matrix of assignments, with nrow as the 
		minimum number of workers to complete the tasks """
		row_sum = min(row_sum, ncols)
		nrows = (col_sum * ncols) / row_sum
		assert(nrows % 1 == 0)
		nrows = max(col_sum, int(nrows))
		M = np.zeros((nrows, ncols))
		for i in range(nrows):
			col_sums = np.sum(M, axis = 0)
			col_min = np.min(col_sums)
			cols = random.sample(list(np.where(col_sums == col_min)[0]), row_sum)
			row_indices = np.array(range(i, i + col_sum))
			M[i, cols] = 1
		return M

	def get_hit_list(self, assignment_matrix = None, start_task_index = 0, perm_tasks1 = [], perm_tasks2 = [], sample_tasks = [], sample_n = 0, taskconditions = None):
		if assignment_matrix is None:
			assignment_matrix = self.get_assignment_matrix(self.assignments_per_task, self.tasks_per_assignment, self.ntasks)
		[nrow, ncol] = assignment_matrix.shape
		tasks_list = [np.where(assignment_matrix[i, ] > 0)[0] for i in range(nrow)]
		perm_tasks1 = " ".join(str(t) for t in perm_tasks1)
		perm_tasks2 = " ".join(str(t) for t in perm_tasks2)
		sample_tasks_list = [" ".join(str(t) for t in random.sample(sample_tasks, sample_n)) for i in range(nrow)]
		if taskconditions is None:
			hit_list = [HIT(i, perm_tasks1 + " " + sample_tasks_list[i] + " " + perm_tasks2 + " " + " ".join(str(start_task_index + task) for task in tasks_list[i])) for i in range(nrow)]
		else:
			hit_list = [HIT(i, perm_tasks1 + " " + sample_tasks_list[i] + " " + perm_tasks2 + " " + " ".join(str(start_task_index + task) for task in tasks_list[i]), 
							taskconditions = [taskconditions[task] for task in tasks_list[i]]) for i in range(nrow)]

		# add exclusions: workers who complete a hit cannot do others with the same tasks
		exclusions = []
		for hit in hit_list:
			exclude = [h.hitid for h in hit_list if len(set(hit.task_list(start_task_index)).intersection(set(h.task_list(start_task_index)))) > 0]
			hit.exclusions = " ".join(str(hitid) for hitid in exclude)
			exclusions.append(exclude)

		return(hit_list)

def wrap_div(content, divclass = "footer"):
	return f"<div class = '{divclass}'>" + content + "</div>"

if __name__ == '__main__':

	label_categories = ['Multimedia News', 'Advertisement', 'Junk', 'Personal', 'Other']

	header = '''
		<head>
		<style type = "text/css">
			.footer {
				position: relative;
				bottom: 5px;
			}
		</style>
		</head>
	'''

	definitions = '''
			<hr>
			<u>Definitions:</u><br>
			<ul>
                <li><b>News Article</b>: A passage from a complete news article.  <i>Includes: national or local news, editorials, audio transcripts, etc.</i></li>
                <li><b>Multimedia News</b>: Text descriptions that accompany news content presented <u>primarily</u> in non-text forms. <i>Includes: audio, image, and video captions, etc.</i></li>
                <li><b>Advertisement</b>: Marketing communications that promote a product, service, or idea. <i>Includes: sign-up and subscription solicitations, spam, etc.</i></li>
                <li><b>Junk</b>: Incomplete or broken content, or webpage-generated messages. <i>Includes: HTML/CSS code, error messages, login and "click to view more" instructions, "More here:" links, cut off sentences, etc.</i></li>
                <li><b>Personal</b>: Non-news content created by individual users. <i>Includes: posts or comments on blogs and forums, etc.</i></li>
                <li><b>Other</b>: Other types of non-news content. <i>Includes: Brief weather or sports game score updates, copyright messages, short author biographies, music or book reviews, etc.</i></li>
            </ul>
	'''

	instructions = '''
			<p>We gathered content from a set of webpages for the purpose of gathering samples of online news articles. You will be shown excerpts of these webpages.
            Your task is to select the label that <b>best</b> describes the original content of each excerpt. That is, you will:</p>
            <p><ol>
                <li>Determine whether the content shown is a valid excerpt from a news article, and</li>
                <li>If not, select the alternative category that <b>best</b> describes the content.</li>
            </ol></li></p>
            <p>If you believe that multiple categories describe the text, please do your best to select what you believe is the predominant category. For instance, it is possible for an article to have a few lines of advertisement or copyright statements within it, but still be a valid news article.</p>
            <p><mark>You will be paid a bonus based on your performance, determined by the degree to which your answers agree with those given by other respondents.</mark></p>
            <p>Below are definitions for each of the possible labels. If this is your first time attempting this HIT, then you will be shown an example article from each category on the following pages.</p>
            <p><i>Please make sure you understand the definitions and examples, as you will need to correctly answer the practice questions before proceeding to the actual task.</i></p>
    '''

	examples = '''
    		<hr>
            <p><b>News Article:</b></p>
            <p>Jun. 1, 2019 12:10 PM EDT<br><br>Tremors rattle southeast Albania, injuring 4, damaging homes<br><br>TIRANA, Albania (AP) â€” An earthquake with a preliminary magnitude of 5.3 struck rural areas in Albania early Saturday southeast of Tirana, the capital, injuring four people and damaging about 100 houses, authorities said.<br><br>The Defense Ministry reported that four people were injured from falling walls at their homes and about 100 houses, many uninhabited, were damaged. The injured were sent to a local hospital.</p>
            <hr>
            <p><b>Multimedia News:</b></p>
            <p>June 01, 2019 08:06 AM<br><br>In this Thursday, May 30, 2019 photo, Brandon "BMike" Odoms, in man-lift, and students in the Young Artist Movement work on a mural in downtown New Orleans. The mural is one of five created as part of an Arts Council of New Orleans project called "Unframed. " The council's executive director, Heidi Schmalbach, says the group wants New Orleans to be known for its contemporary art as much as for its music, food and culture. Janet McConnaughey AP Photo</p>
            <hr>
            <p><b>Advertisement:</b></p>
            <p>Take Five - This is your final free article during this 30 day period. Stay in touch with all of the news. Sign up today for complete digital access to The Daily News-Record.<br><br><br><br>Sign up today for complete digital access to The Daily News-Record.<br><br>Thank you for signing in!<br><br><br><br>This is your fourth of five free articles during this 30 day period.</p>
            <hr>
            <p><b>Junk:</b></p>
            <p>FacebookEmail Twitter Google+ LinkedIn Pinterest<br><br>A 'woman of God,' a Scottish music lover: These are the victims of Virginia Beach shooting<br><br>Virginia Beach Police said 12 people were killed in the mass shooting and at least four more were wounded. An officer was among those injured.<br><br>A 'woman of God,' a Scottish music lover: These are the victims of Virginia Beach shooting Virginia Beach Police said 12 people were killed in the mass shooting and at least four more were wounded. An officer was among those injured. Check out this story on timesrecordnews.com: https://www.usatoday.com/story/news/nation/2019/06/01/virginia-beach-mass-shooting-victims-names-released-others-injured/1310647001/</p>
            <hr>
            <p><b>Personal:</b></p>
            <p>Isn't CSGO sort of arcade gunplay too? Perhaps I don't understand your definition of the term.<br><br><br><br><br><br><br><br><br><br>Yes it does sound reasonable, also it is more appealing because would've been an open world "Bioshock", but they wanted to much on one game and couldn't decide which one should stay.<br><br>Theoretically you are correct, if a big studio or publisher can see the potential in it, hope fully picks up and do something good, but nowadays I barely see any good game which is not:<br><br>- a Beta version released as Full game and 2 years later (when no one cares) become the one which should've been the version they start with.<br><br>- in it's Alpha state released as Full game and reach Beta in the next 1-2 years as a Full Game</p>
            <hr>
            <p><b>Other</b></p>
            <p>Partly cloudy starting in the afternoon, continuing until evening.<br><br>13% chance of precipitation<br><br>Partly cloudy starting in the afternoon.<br><br>17% chance of precipitation<br><br>Mostly cloudy throughout the day.<br><br>19% chance of precipitation</p>
            <hr>
	'''
	examples_list = ['''<p><b>News Article:</b></p><p>Jun. 1, 2019 12:10 PM EDT<br><br>Tremors rattle southeast Albania, injuring 4, damaging homes<br><br>TIRANA, Albania (AP) â€” An earthquake with a preliminary magnitude of 5.3 struck rural areas in Albania early Saturday southeast of Tirana, the capital, injuring four people and damaging about 100 houses, authorities said.<br><br>The Defense Ministry reported that four people were injured from falling walls at their homes and about 100 houses, many uninhabited, were damaged. The injured were sent to a local hospital.</p>''', 
					'''<p><b>Multimedia News:</b></p><p>June 01, 2019 08:06 AM<br><br>In this Thursday, May 30, 2019 photo, Brandon "BMike" Odoms, in man-lift, and students in the Young Artist Movement work on a mural in downtown New Orleans. The mural is one of five created as part of an Arts Council of New Orleans project called "Unframed. " The council's executive director, Heidi Schmalbach, says the group wants New Orleans to be known for its contemporary art as much as for its music, food and culture. Janet McConnaughey AP Photo</p>''',
            		'''<p><b>Advertisement:</b></p><p>Take Five - This is your final free article during this 30 day period. Stay in touch with all of the news. Sign up today for complete digital access to The Daily News-Record.<br><br><br><br>Sign up today for complete digital access to The Daily News-Record.<br><br>Thank you for signing in!<br><br><br><br>This is your fourth of five free articles during this 30 day period.</p>''',
            		'''<p><b>Junk:</b></p><p>FacebookEmail Twitter Google+ LinkedIn Pinterest<br><br>A 'woman of God,' a Scottish music lover: These are the victims of Virginia Beach shooting<br><br>Virginia Beach Police said 12 people were killed in the mass shooting and at least four more were wounded. An officer was among those injured.<br><br>A 'woman of God,' a Scottish music lover: These are the victims of Virginia Beach shooting Virginia Beach Police said 12 people were killed in the mass shooting and at least four more were wounded. An officer was among those injured. Check out this story on timesrecordnews.com: https://www.usatoday.com/story/news/nation/2019/06/01/virginia-beach-mass-shooting-victims-names-released-others-injured/1310647001/</p>''',
            		'''<p><b>Personal:</b></p><p>Isn't CSGO sort of arcade gunplay too? Perhaps I don't understand your definition of the term.<br><br><br><br><br><br><br><br><br><br>Yes it does sound reasonable, also it is more appealing because would've been an open world "Bioshock", but they wanted to much on one game and couldn't decide which one should stay.<br><br>Theoretically you are correct, if a big studio or publisher can see the potential in it, hope fully picks up and do something good, but nowadays I barely see any good game which is not:<br><br>- a Beta version released as Full game and 2 years later (when no one cares) become the one which should've been the version they start with.<br><br>- in it's Alpha state released as Full game and reach Beta in the next 1-2 years as a Full Game</p>''',
            		'''<p><b>Other</b></p><p>Partly cloudy starting in the afternoon, continuing until evening.<br><br>13% chance of precipitation<br><br>Partly cloudy starting in the afternoon.<br><br>17% chance of precipitation<br><br>Mostly cloudy throughout the day.<br><br>19% chance of precipitation</p>'''
					]

	print("Creating Article Filtering Task...")
	scrape_file = "mturk_scrapes_20190601.csv"
	#scrape_file = "mturk_scrapes_sample.csv"
	input_csv = pd.read_csv(scrape_file)
	screening_csv = pd.read_csv("mturk_scrapes_sample_KEY.csv")

	nscreening = min(5, screening_csv.shape[0])
	ntasks = input_csv.shape[0]
	assignments_per_task = 3
	tasks_per_assignment = 25
	print(f"{ntasks} tasks, {assignments_per_task} assignments per task, {tasks_per_assignment} tasks per assignment")

	## QUESTIONS ##
	q_intro = Question(varname = "intro", 
					questiontext = "Please take time to read the instructions carefully before you proceed.",
					valuetype = "categorical", 
					categories = ["I have read and understood the task instructions."])

	q_example = Question(varname = "example", 
					questiontext = "Please take time to understand why the example label is appropriate before you proceed.",
					valuetype = "categorical", 
					categories = ["I have read and understand the label choice for this example."])

	qlist_valid_screen = [Question(varname = f"screen_valid{i}", 
					questiontext = "Is this a passage from a valid news article?",
					valuetype = "categorical", 
					categories = ["Yes", "No"], 
					values = [2, 1] if screening_csv["answer"][i] == "Valid" else [0, 1], 
					helptext = "A valid news article is a text-based article reporting on news.") for i in range(screening_csv.shape[0])] # Add screening questions 

	qlist_label_screen = [Question(varname = f"screen_label{i}", condition = f"screen_valid{i}==1",
					questiontext = "Select the label that best describes the content shown.",
					valuetype = "categorical",
					categories = label_categories, 
					values = [1 if cat == screening_csv["answer"][i] else 0 for cat in label_categories],
					helptext = "Please pick the best answer based on the provided definitions of each category") for i in range(screening_csv.shape[0])]

	q_understand = Question(varname = "pass_screen", 
				questiontext = "Please find and check the correct labels to the practice questions, shown to the left.",
				valuetype = "categorical", 
				categories = ["I have checked and understood the practice labels."])

	q_valid = Question(varname = "valid",
					questiontext = "Is this a passage from a valid news article?",
					valuetype = "categorical", 
					categories = ["Yes", "No"],
					helptext = "A valid news article is a text-based article reporting on news. Please try your best; you will be paid based on the accuracy of your labels!",
					bonus = ["threshold:50", 1])

	q_label = Question(varname = "label", condition = "valid==No",
					questiontext = "Select the label that best describes the content shown.",
					valuetype = "categorical",
					categories = label_categories, 
					helptext = "Please pick the best answer based on the provided definitions of each category. Please try your best; you will be paid based on the accuracy of your labels!",
					bonus = ["threshold:50", 1])

	## MODULES ## 
	module_intro = Module(name = "intro", header = "Read the instructions carefully.", questions = [q_intro])
	module_example = Module(name = "example", header = "Examples", questions = [q_example])
	modlist_screen = [Module(name = f"screen{i}", header = "Practice: Choose the best answer based on the definitions provided.", questions = [qlist_valid_screen[i], qlist_label_screen[i]]) for i in range(screening_csv.shape[0])]
	module_pass_screen = Module(name = "pass_screen", header = "End of practice rounds", questions = [q_understand])
	module_label = Module(name = "label", header = "Choose the best answer.", questions = [q_valid, q_label])

	## TASKS ##
	# Note: taskID's 0 - 99 are reserved for non-label tasks.
	tasklist_instructions = [Task(name = "intro", taskid = 0, modules = module_intro, 
							content = header + instructions + definitions)] + [
							Task(name = f"example{i}", taskid = i + 1, modules = module_example, 
							content = header + examples_list[i] + definitions) for i in range(len(examples_list))]

	example_condition = "notinset{$workerid,exclude_examples}"

	# Screening tasks: check for correct answers 
	tasklist_screen = [Task(name = f"screen_task{i}", taskid = i + len(examples_list) + 1, modules = modlist_screen[i], 
							content = header + screening_csv["html"][i] + wrap_div(definitions)) for i in range(screening_csv.shape[0])]

	labels_pass_screen = [screening_csv["answer"][i] for i in range(screening_csv.shape[0])]
	doc_pass_screen = "<hr>".join("<b>" + labels_pass_screen[i] + "</b><br>" + screening_csv["html"][i] for i in range(screening_csv.shape[0]))
	tasklist_pass_screen = [Task(name = "pass_screen", taskid = screening_csv.shape[0] + len(examples_list) + 1, modules = module_pass_screen, 
							content = header + doc_pass_screen + definitions)]
	
	screen_condition = "+".join(f"{i + len(examples_list) + 1}*screen{i}*screen_valid{i}+{i + len(examples_list) + 1}*screen{i}*screen_label{i}" for i in range(screening_csv.shape[0]))
	screen_condition += f" >= {int(nscreening + nscreening/2)}"
	print("Screening condition for later tasks: ", screen_condition)

	tasklist_label = [Task(name = f"label_task{i}", taskid = i + 100, modules = module_label,
						   content = header + str(input_csv.loc[i, "html"]) + wrap_div(definitions)) for i in range(ntasks)]
	input_csv["task"] = [i + 100 for i in range(ntasks)]
	input_csv.to_csv(scrape_file, index = False)

	## SETS ## 
	set_exclude_examples = Set(name = "exclude_examples", members_list = ["admin", "ling"])
	
	## XML DOC, DOCUMENTS ## 
	XMLdoc = XMLdoc(modules = [module_intro, module_example] + modlist_screen + [module_pass_screen] + [module_label],
					tasks = tasklist_instructions + tasklist_screen + tasklist_pass_screen + tasklist_label, 
					hits = Survey(ntasks, assignments_per_task, 
								tasks_per_assignment).get_hit_list(start_task_index = 100, 
								perm_tasks1 = [t for t in range(len(examples_list) + 1)],
								perm_tasks2 = [len(examples_list) + screening_csv.shape[0] + 1], 
								sample_tasks = [t for t in range(len(examples_list) + 1, len(examples_list) + screening_csv.shape[0] + 1)], 
								sample_n = nscreening))
	XMLdoc.add_documents([t.document for t in XMLdoc.tasks])
	XMLdoc.add_sets([set_exclude_examples])

	for hit in XMLdoc.hits:
		for tid in range(1, len(examples_list) + 1):
			hit.add_taskcondition(tid, example_condition)
		for tid in hit.task_list(start_task_index = 100):
			hit.add_taskcondition(tid, screen_condition)

	print(f"{len(XMLdoc.hits)} total hits created")
	
	XMLdoc.write_xml("sample_survey.xml")
