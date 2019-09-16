import numpy as numpy
import pandas as pd

class Question:
	"""
	<question>
	  <varname>internal_variable_name</varname>
	  <questiontext>Visible question text</questiontext>
	  (<helptext>Optional help text</helptext>)
	  <valuetype>some_value_type</valuetype>
	  ...
	</question>
	"""
	def __init__(self, varname, valuetype, questiontext, categories = None, values = None,
				 options = None, condition = None, bonus = None, helptext = None):
		self.q_dict = None 
		self.varname = varname # used when recording responses 
		# valuetypes = ['numeric', text', 'categorical']
		self.valuetype = valuetype # determines how question is rendered 
		self.questiontext = questiontext 
		self.categories = categories
		self.values = values
		self.options = options # for use with scaled categorical questions 
		# condition format: varname [==|!=] value 
		self.condition = condition 
		# Bonus should be specified as: [scheme, points]
		# Where schemes = ['linear', 'threshold:50']
		self.bonus = bonus 
		self.helptext = helptext 

	def to_dict(self):
		self.q_dict = {"varname": self.varname, 
					   "questiontext": self.questiontext, 
					   "valuetype": self.valuetype}

		if self.options is not None:
			self.q_dict["options"] = self.options 

		if self.condition is not None:
			self.q_dict["condition"] = "<![CDATA[" + self.condition + "]]>"

		if self.bonus is not None:
			self.q_dict["bonus"] = self.bonus[0] 
			self.q_dict["bonuspoints"] = self.bonus[1] 

		if self.helptext is not None:
			self.q_dict["helptext"] = self.helptext

		if self.valuetype == "categorical":
			assert self.categories is not None 
			self.q_dict["content"] = {"categories": self.categories_dict(self.categories, self.values)}

		return self.q_dict

	def categories_dict(self, categories, values = None):
		if values is None:
			values = categories
		categories_dict = [Category(categories[c], values[c]).to_dict() for c in range(len(categories))]
		return categories_dict

class Category:
	def __init__(self, text, value):
		self.cat_dict = None
		# For nexted categorical questions, use "|" between options in text
		self.text = text 
		self.value = value 

	def to_dict(self):
		self.cat_dict = {"text": self.text, 
						 "value": self.value}
		return self.cat_dict
		

class Options:
	""" Note: unlike other classes, this returns a list of dicts, not a single dict 
	"""
	def __init__(self, layout = "horizontal", lowLabel = None, highLabel = None,
				 outsideCategories = None):
		self.options_list = None
		self.layout = layout
		self.lowLabel = lowLabel
		self.highLabel = highLabel
		self.outsideCategories = outsideCategories

	def to_list(self):
		self.options_list = [{"layout": self.layout}]
		if self.lowLabel is not None:
			self.options_list.append({"lowLabel": self.lowLabel})
		if self.highLabel is not None:
			self.options_list.append({"highLabel": self.highLabel})
		if self.outsideCategories is not None:
			self.options_list.extend([{"outsideCategories": oc} for oc in self.outsideCategories])
		return self.options_list