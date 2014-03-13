#! /usr/bin/python2
# vim: set fileencoding=utf-8
from dateutil.parser import parse
from subprocess import check_output
from shutil import copy
import datetime
import sys
import os.path
import isoweek
DATE_FORMAT = '%Y%m%d'
START = """\documentclass[a4paper,oneside,draft,
notitlepage,11pt,svgnames]{scrreprt}
\\newcommand{\workingDate}{\\today}
\input{preambule}
\\begin{document}
"""
END = """
\printbibliography{}
\end{document}"""
MD_ACTIVITY = """# Activity {.unnumbered}

~~~~
"""


def create(date):
    filename = date.strftime(DATE_FORMAT)
    month = date.strftime('%B')
    day = date.strftime('%d')
    with open('template.tex', 'r') as t:
        content = t.read()
        content = content.replace('MONTH', month)
        content = content.replace('DAY', day)
        content = content.replace('content', filename+'.tex')
    with open('current.tex', 'w') as f:
        f.write(content)
    copy('content.md', filename+'.md')
    print('gvim {}'.format(filename+'.md'))


def week(date):
    week = isoweek.Week.withdate(date)
    name = 'w{}.tex'.format(week.week)
    together([week.day(d) for d in range(7)], name)


def together(dates, name):
    include = '\chapter{{{}}}\n\input{{{}}}'
    res = [include.format(d.strftime('%B %d'),
                          d.strftime(DATE_FORMAT)) for d in dates
           if os.path.exists(d.strftime(DATE_FORMAT)+'.tex')]
    with open(name, 'w') as f:
        f.write(START+'\n'.join(res)+END)
    print('mv {} w.tex'.format(name))


def log(date):
    cmd = "git whatchanged --since='{}' --pretty=format:'%B'"
    cmd += "|sed '/^$/d'|sed 's/^.*\.\.\. //'"
    since = date.replace(hour=4)
    log = check_output(cmd.format(str(since)),
                       shell=True).strip()+"\n\n~~~~"
    print(log)
    return log.replace('\t', '    ')


def since(date):
    today = datetime.datetime.now()
    name = date.strftime(DATE_FORMAT) + '_' + today.strftime(DATE_FORMAT)
    days = [(date + datetime.timedelta(days=i)).date()
            for i in range(1, (today-date).days+1)]
    together(days, name+'.tex')


def finish(date):
    today = datetime.datetime.now()
    name = today.strftime(DATE_FORMAT)
    with open(name+'.md', 'a') as f:
        f.write(MD_ACTIVITY+log(today))
    cmd = 'pandoc -f markdown -t latex {}.md'
    cmd += " |grep -v addcontent|sed -e '/^\\\\sec/ s/\\\\label.*$//'"
    print(cmd.format(name))
    latex = check_output(cmd.format(name), shell=True)
    with open(name+'.tex', 'w') as today_log:
        today_log.write(latex)
    print('latexmk -pdf -pvc current')
    print('mv current.pdf {}.pdf'.format(name))

if __name__ == '__main__':
    date = datetime.datetime.now()
    command = 'create'
    if len(sys.argv) > 1:
        command = sys.argv[1].strip()
        if len(sys.argv) > 2:
            date = parse(sys.argv[2], dayfirst=True)

    globals()[command](date)
