"""This file is responsible for generating the XML file used with the 
crowdsourcer library to generate mturk HIT's according to the following format:

<xml>
  <modules>
    ... module definitions ...
  </modules>
  <tasks>
    ... task definitions ...
  </tasks>
  <hits>
    ... hit definitions ...
  </hits>
  <documents>
    ... document definitions ...
  </documents>
</xml>

(see https://github.com/markusmobius/news_crowdsourcer for documentation)
"""
import numpy as numpy
import pandas as pd
import html
import dicttoxml 
from xml.dom.minidom import parseString

# Import individual XML item classes
from Question import Question, Category, Options

class XMLdoc:
	items = {"modules":"module", "questions": "question", "categories":"category",
				"documents":"document", "tasks":"task", "hits": "hit", 
				"taskconditions": "taskcondition", "content": "categories", "sets": "set"}

	def __init__(self, modules = None, tasks = None, hits = None, sets = None, documents = None):
		self.xml_dict = None
		self.modules = [] if modules is None else modules
		self.tasks = [] if tasks is None else tasks
		self.hits = [] if hits is None else hits
		self.sets = sets
		self.documents = [] if documents is None else documents # Optional 

	def write_xml(self, file, toprettyxml = True):
		with open(file, "w", encoding="utf-8") as xmlfile:
			xmlfile.write(self.to_xml(toprettyxml))

	def to_xml(self, toprettyxml = True):
		self.xml_dict = self.to_dict()
		rename_subitems = lambda x: self.items[x] if x in self.items else x 
		xml = dicttoxml.dicttoxml(self.xml_dict, custom_root = 'xml', 
												 item_func = rename_subitems, 
												 attr_type = False)

		#with open("temp.txt", "w", encoding = "utf-8") as f:
		#	f.write(xml.decode().encode("ascii", "xmlcharrefreplace").decode("utf-8"))

		if not toprettyxml:
			return xml
		xml = parseString(xml).toprettyxml()
		xml = xml.replace("&lt;![CDATA[", "<![CDATA[").replace("]]&gt;", "]]>")
		xml = html.unescape(xml)
		return xml

	def to_dict(self):
		self.xml_dict = {"modules": [m.to_dict() for m in self.modules], 
						"tasks": [t.to_dict() for t in self.tasks], 
						"hits": [h.to_dict() for h in self.hits], 
						"documents": [d.to_dict() for d in self.documents]}
		if self.sets is not None:
			self.xml_dict["sets"] = [s.to_dict() for s in self.sets]

		return self.xml_dict

	def add_modules(self, modules):
		self.modules.extend(modules)

	def add_tasks(self, tasks):
		self.tasks.extend(tasks)

	def add_hits(self, hits):
		self.hits.extend(hits)

	def add_documents(self, documents):
		self.documents.extend(documents)

	def add_sets(self, sets):
		if self.sets is None:
			self.sets = []
		self.sets.extend(sets)


class Module:
	"""
	<module>
	  <name>module_name</name>
	  <header>Visible Module Header</header>
	  <questions>
	    ... question definitions ...
	  </questions>
	</module>
	"""
	def __init__(self, name, header, questions = None):
		self.module_dict = None
		self.name = name
		self.header = header
		self.questions = questions 

	def to_dict(self):
		self.module_dict = {"name": self.name, 
							"header": self.header,
							"questions": [q.to_dict() for q in self.questions]}
		return self.module_dict 


class Task:
	def __init__(self, name, taskid, modules, content = None):
		self.task_dict = None
		self.document = Document(name, content)
		self.content = self.document.name # html document name
		self.taskid = taskid
		self.modules = modules 


	def to_dict(self):
		self.task_dict = {"content": self.content, 
						  "taskid": self.taskid}

		if type(self.modules) is list:
			self.task_dict["modules"] = " ".join(mod.name for mod in self.modules)
		else: 
			self.task_dict["modules"] = self.modules.name
		return self.task_dict 

class Document:
	default_content = "<![CDATA[" + "<p>Please answer the questions shown.</p>" + "]]>"
	default_content = "Please answer the questions shown."

	def __init__(self, name, content = None):
		self.doc_dict = None 
		self.name = name + ".html"
		self.content = content if content is not None else self.default_content 

	def clean_content(self, content):
		# not used right now
		content = content.encode('ascii', 'xmlcharrefreplace').decode()
		return content

	def to_dict(self):
		self.doc_dict = {"name": self.name,
						"content": "<![CDATA[" + self.content  + "]]>"}

		return self.doc_dict 

class HIT:
	def __init__(self, hitid, tasks, exclusions = None, taskconditions = None):
		self.hit_dict = None
		self.hitid = hitid
		self.tasks = tasks
		self.exclusions = exclusions
		self.taskconditions = taskconditions

	def to_dict(self):
		self.hit_dict = {"hitid": self.hitid, 
						 "tasks": self.tasks}

		if self.exclusions is not None:
			self.hit_dict["exclusions"] = self.exclusions

		if self.taskconditions is not None:
			self.hit_dict["taskconditions"] = self.taskconditions

		return self.hit_dict

	def add_taskcondition(self, taskid, condition):
		if self.taskconditions is None:
			self.taskconditions = []
		self.taskconditions.append({"taskid": taskid,
					"condition": "<![CDATA[" + condition + "]]>"})

	def task_list(self, start_task_index = 0):
		task_list = self.tasks.split()
		return [i for i in task_list if int(i) >= start_task_index]

class Set:
	def __init__(self, name, members_list = None):
		self.set_dict = None
		self.name = name
		self.members_list = members_list if members_list is not None else []
		self.members = None
		self.update_members()

	def to_dict(self):
		self.set_dict = {"name": self.name,
						"members": self.members}
		return self.set_dict

	def add_members(self, members_list):
		self.members_list += members_list
		self.update_members()

	def delete_member(self, member):
		if member in self.members_list:
			self.members_list.remove(member)
		self.update_members()

	def update_members(self):
		self.members = " ".join(self.members_list)

	def reset(self):
		self.members_list = []
		self.members = None
