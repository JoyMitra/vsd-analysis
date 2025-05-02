# vsd-analysis
The repository contains scripts to reproduce vsd analysis data

## Collect Issues

Run the Python script `src/download-issues.py -h` to see how to run the script. The `input` directory has json files. Each json file has an array of repo names.

## Stakeholders

The following UNIX commands will reproduce the number of stakeholders from a collection of issues. The commands were run on a Max OSX machine.

1. get all descriptions from a collection of GitHub issues. The descriptions contain information about stakeholders. We assume that all issues of a repo are in a txt file and the repo has the name
of the form *final-project-REPONAME*

`$ grep -i -A8 "description" /path/to/issues/final-project-/*.txt* > analysis/<descfile>`

2. Get lines that have a potential stakeholder. We assume that all stakeholders must be contained within the lines "As a <stakeholder> ... ".

`$ grep -i 'as a' analysis/<descfile> > /path/to/analysis/outFile`

3. get unique stakeholders from file generated in the previous step. 

`$ grep -o -i "as a \w*\b\|\bas a [[:alnum:][:space:]_-]*," /path/to/analysis/outFile | sed -E 's/as a //i' | sed -E 's/,$//' | sort | uniq > /path/to/analysis/stakeholderFile`

4. Some stakeholder terms may not make sense. You can search for the words in `/path/to/analysis/outFile` to understand the context in which it was defined and whether you should consider it.

## Values

First extract the context for each issue using the following command:

`$ python src/extract-context.py /path/to/issues /path/to/output`

Use the following command to run topic modeling on the files generated in the previous step. The number of topics assumed is 7 as that is the optimal based on the topic distribution

`python src/topic-modeling.py /path/to/contextfiles --output_dir /path/to/out --n_topics <INTEGER>`