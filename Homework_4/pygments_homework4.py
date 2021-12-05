#!/usr/bin/env python

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

file_list = ['me570_graph.py', 'me570_geometry.py', 'me570_robot.py']
file_out_basename = 'homework4'

# Initialize contents of the final HTML file with the HTML header
contents = [
    '<!DOCTYPE html>', '<html lang="en">', '<head>',
    '<link rel="stylesheet" type="text/css" href="' + file_out_basename +
    '.css">', '</head>', '<body>'
]

# Pygments objects
lexer = PythonLexer()
formatter = HtmlFormatter()

# Add the formatted contents of each py file with a title
for file_name in file_list:
    contents += [
        '<h1>' + file_name + '</h1>',
    ]
    with open(file_name, 'rt') as f_in:
        formatted_code = highlight(f_in.read(), lexer, formatter)
        contents += [formatted_code]

# Close the HTML
contents += ['</body>', '</html>']

# Write out the HTML and CSS files
with open(file_out_basename + '.html', 'wt') as f_out:
    f_out.write('\n'.join(contents))
    print('Generated', file_out_basename + '.html', 'containing', file_list)

with open(file_out_basename + '.css', 'wt') as f_out:
    f_out.write(formatter.get_style_defs())
    print('Generated', file_out_basename + '.css')
