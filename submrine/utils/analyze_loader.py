import sys
import os

from nibabel.fileholders import FileHolder
from nibabel.analyze import AnalyzeImage


def load_image(analyze_img_path, analyze_header_path):
	"""
	Loads an MRI image from a stored representation
	in Analyze 7.5 format

	Parameters
	------------
	analyze_img_path : str
		The path to the Analyze image path (with ".img" extension)
	analyze_header_path : str
		The path to the Analyze header path (with ".hdr" extension)

	Returns
	------------
	nibabel.analyze.AnalyzeImage
		An nibabel image object representing the Analyze 7.5
		MRI image specified by the provided paths
	"""

	img_holder = FileHolder(analyze_img_path)
	header_holder = FileHolder(analyze_header_path)

	data_map = {'image' : img_holder, 'header' : header_holder}
	return AnalyzeImage.from_file_map(data_map, mmap=False)

def load_image_data(analyze_img_path, analyze_header_path):
	"""
	Loads an MRI image from a stored representation
	in Analyze 7.5 format

	Parameters
	------------
	analyze_img_path : str
		The path to the Analyze image path (with ".img" extension)
	analyze_header_path : str
		The path to the Analyze header path (with ".hdr" extension)

	Returns
	------------
	np.ndarray
		A numpy array representing the image's underlying data.
		This is a result of a call to `nibabel.analyze.AnalyzeImage.get_data()`
	"""

	return load_image(analyze_img_path, analyze_header_path).get_data()

