import argparse


def main():
	"""
	This function sets the global variables
	"""
	parser = argparse.ArgumentParser(
									formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data_dir', type=str, default='data/',
						help='data directory containing images')

	args = parser.parse_args()
	train(args)

def train(args):
	"""
	This function performs the training
	"""
	print("arg dict is: ", args)



main()