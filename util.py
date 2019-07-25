#===========================
#!/usr/bin/python
#utility functions
#===========================
import sys, collections, fnmatch, copy, os, json
class cfg:
	def __init__(self, ):
		self.datadir=''
		self.paraviewdir=''
	def parseConfig(self,configure):
		with open(configure, 'r') as cf:
		    config = json.load(cf)
		self.datadir=os.environ.get('HOME')+config['DIRECTORY']['DATA_DIR']
		self.paraviewdir=os.environ.get('HOME')+config['DIRECTORY']['PARAVIEW_DIR']
		return self
