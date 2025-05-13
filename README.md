# Context

This project was motivated by the need to understand and measure the
effectiveness of teaching value sensitive design in a graduate-level 
software engineering course at Northeastern University. Students in the course
worked on a team-based project to design and implement a mock or fake 
stackoverflow web application using value sensitive design (VSD). The 
course was taught in Fall 2024 and Spring 2025. In Fall 2024, students 
were explicitly taught VSD but in Spring 2025 they were not. This design
was intentional as we wanted to measure the differences in how students
think about the principles of VSD when they are explicitly taught about 
them vs. when they are not. 

In both semester, students defined their requirements as user stories and
documented the stories in GitHub Issues. Hence, we extracted the data 
in those issues and analyzed them. The following is a description of 
the directories in this repository:

- The **src** directory has all the necessary Python scripts. You will
need to have Python 3.10 or higher installed and setup to run the scripts. 
You only need to run the scripts if you need to reproduce the data needed
for the analysis, otherwise you can ignore this directory. 

- The **issues-fa24** and **issues-sp25** directories contain the user stories
defined by each collection of students. Each _issue_ directory has a 
bunch of subdirectory of the form _final-project-teamname_. Each subdirectory
has a bunch of _txt_ files. Each such _txt_ file is a user story that 
captures the description, the socio-historical context, and the scenarios
used to derive the test cases.

    If you do not see the _issues_ directory then you can generate it by 
following the instructions in **Collect Issues** section. 

- The **context-fa24** and the **context-sp25** directories only the
socio-historical context of a user story as defined by the team. Each 
_context_ directory has a bunch of subdirectorties for each team in that
semester. Each such subdirectory has a collection of txt files with the 
context information. 

    If you do not find these directories follow the instructions in the 
**Value Extraction** section to generate them.

- The **analysis** directory contains the **stakeholder** files. Each such
file contains the unique stakeholders identified by the teams in each semester. If you do not see these files, you can generate them by
following the instructions in the **Stakeholder Extraction** section. 

    Further, this directory contains the **values** directory for 
each semester. The **values* directory contain the results of running
the topic modeling script on the _context_ directories. You can view the 
results by opening the summary files in a browser (e.g., values-sp25/summary_report.html). The summary files gives us a collection of 
word clouds or topics which can be used to identify the broad themes
covered by the user stories in each semester. 

## Collect Issues

You will need to create your own GitHub API key to use the script. 

Run the Python script `src/download-issues.py -h` to see how to run the script. The `input` directory has json files. Each json file has an array of of repo names. 

## Stakeholder Extraction

The following UNIX commands will reproduce the number of stakeholders from a collection of issues. The commands were run on a Max OSX machine.

1. get all descriptions from a collection of GitHub issues. The descriptions contain information about stakeholders. We assume that all issues of a repo are in a txt file and the repo has the name
of the form *final-project-REPONAME*

`$ grep -i -A8 "description" /path/to/issues/final-project-/*.txt* > analysis/<descfile>`

2. Get lines that have a potential stakeholder. We assume that all stakeholders must be contained within the lines "As a <stakeholder> ... ".

`$ grep -i 'as a' analysis/<descfile> > /path/to/analysis/outFile`

3. get unique stakeholders from file generated in the previous step. 

`$ grep -o -i "as a \w*\b\|\bas a [[:alnum:][:space:]_-]*," /path/to/analysis/outFile | sed -E 's/as a //i' | sed -E 's/,$//' | sort | uniq > /path/to/analysis/stakeholderFile`

4. Some stakeholder terms may not make sense. You can search for the words in `/path/to/analysis/outFile` to understand the context in which it was defined and whether you should consider it.

## Value Extraction

First extract the context for each issue using the following command:

`$ python src/extract-context.py /path/to/issues /path/to/contextfiles`

Use the following command to run topic modeling on the files generated in the previous step. The number of topics assumed is 7 as that is the optimal based on the topic distribution

`python src/topic-modeling.py /path/to/contextfiles --output_dir /path/to/out --n_topics <INTEGER>`