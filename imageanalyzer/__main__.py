#!/usr/bin/env python

from warnings import filterwarnings
filterwarnings('ignore', category=RuntimeWarning, module='runpy')

def main_logic():
	from argparse import ArgumentParser, SUPPRESS

	class _ArgumentParser(ArgumentParser):
		def error(self, message):
			self.exit(2, 'Error: %s\n' % (message))

	ARG_PARSER = _ArgumentParser(description='Qeeqbox/image-analyzer', usage=SUPPRESS)
	ARG_PARSER._action_groups.pop()
	ARG_PARSER_DRIVERS = ARG_PARSER.add_argument_group('Temp')
	ARG_PARSER_DRIVERS.add_argument('--test', help='Test if project works or not', metavar='', default='')
	ARGV = ARG_PARSER.parse_args()
	print("Please use this package as an object")

if __name__ == '__main__':
    main_logic()