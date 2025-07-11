[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/qXX-HXO8)
# ST446 course project
(revised WT 2025)

## Project task

You are free to propose a task for your project. Your project should demonstrate that you have acquired a good knowledge of some of the topics on distributed computing for big data covered in the course, with a focus on methodology, distributed computing algorithms and technologies for processing big data, and demonstrating performance gains that can be achieved by using distributed computing.

Your implementation would typically be using Spark and/or possibly some other system, and a dataset suitable for your problem.

Your proposal must be approved by the course lecturer. Your proposal may either be immediately approved or you may be asked to revise and resubmit your proposal.

<hr>

## Past projects and suggested topics

* [Here](./Project-examples.md) you will find information about past project topics and links to some references.
* You may also consider [this notebook](./MLprojects.ipynb) from Triguero \& Galar's book with ideas for **problem-based** and **technique-based** projects. Some project ideas have a GitHub repository with reference code.

<hr> 

## Group work

Your project is a group project, with each group consisting of 3 or 4 students.

You are given the freedom to form groups as you like. We will provide some means to facilitate group formation.

You are expected to split the work on your project among yourself. It is expected that each group member provides a fair share of technical contributions to the project.

<hr> 

## GitHub

You will be assigned a private repository for your project in the ST446 organisation.

All solution files (main code and auxiliary scripts), project report and statement of individual contributions must be included in this repository. **Datasets do not need to be uploaded but you must provide the links and/or references to where they can be accessed** and any relevant information related to data acquisition as part of your report.

<hr> 

## Report

### Format of the report

Your report should be formatted according to the [ACM SIG conference proceedings](https://www.overleaf.com/latex/templates/association-for-computing-machinery-acm-sig-proceedings-template/bmvfhcdnxfty) (two-column format).

Your report should be no longer than **eight** pages, not including references, plus unlimited space for references. Your report may have an appendix containing further details of your work.

### Organisation of the report

The layout of the sections of your report should follow the standard structure of a research paper. Here is an example of a research paper organisation:

* **Abstract** The abstract should contain a summary of your report, covering your chosen task, what is your solution, and what are your main results.

* **Introduction** The introduction should give a clear description of your chosen task, why your task is important, what is your solution, why is your solution an appropriate solution for the given problem, and a summary of the main results.

* **Related work** This section should overview related work for the task that you are trying to solve and position your work against related work. Related work would typically be published in research papers, but you can also refer to technical reports related to your chosen problem and tools.

* **Methodology** This section should describe the underlying distributed computing methodology deployed for your task. You may consider structuring the section to cover a standard pipeline of data acquisition and preparation, exploratory data analysis (when pertinent), model training and evaluation, and any interpretation and/or visualisation of results. You are **not** expected to explain your code, but your main methodological decisions and assumptions/constraints present in your solution. We are interested in how have you structured your solution in terms of libraries and APIs, data in/out, and specific (technical) aspects of your solution. You may consider some aspects of reproducibility when structuring your solution, so it would be adaptable to other datasets or models.

* **Numerical results** This section should present your numerical evaluation results along with a discussion of the results. First, the goals of the numerical evaluations should be clearly described. Second, the datasets used in the evaluation should be described. Third, any evaluation metrics used in this section should be defined. Fourth, any baseline methods used for comparison should be briefly described. Fifth, and last, numerical results should be presented along with discussion. You may consider metrics directly related to your chosen problem (for instance, evaluating machine learning models), along with metrics related to distributed computing, such as memory and disk usage, number of partitions/machines vs dataset size, execution time and other scalability metrics.

* **Conclusion** The conclusion section should summarise the content of the report, followed by a discussion of potential future work to address any limitations of the study or to explore new research avenues.

* **Bibliography** The bibliography lists references cited in your report (it is automatically generated by Latex). Your report must cite any references that you used in your research, and give proper credit for any concepts or results introduced in previously published work.

At the end of the report, you should put two statements:
* **Statement about individual contributions**, in which you need to summarise the individual technical contributions of each group member. This can be a Gantt chart or a spreadsheet stating the percentual (%) contribution of each group member towards all tasks described in the methodology (0% in case of no contribution to a specific task). All contributions per task should add to 100%.
* **Statement about the use of generative AI and chat logs**, in which you should disclose any use of AI tools as per departmental and course policies, and upload any chat logs. Please, check the guidelines on [Moolde](https://moodle.lse.ac.uk/course/view.php?id=5824).

<hr>

### Writing tips

Here are some resources on writing research papers that you may find useful for this course but also more generally:

* Jean-Yves Le Boudec, [How to Write a Paper](https://leboudec.github.io/leboudec/resources/paper.html), accessed 2025
* John N. Tsitsiklis, [A Few Tips on Writing Papers with Mathematical Content](http://web.mit.edu/jnt/www/Papers/R-20-write-v5.pdf), last update 2020, accessed 2025
* Jon Turner, [How to Write a Great Research Paper](https://www.arl.wustl.edu/~pcrowley/cse/591/writingResearchPapers.pdf), accessed 2025

<hr>

### Marking

Marking criteria are defined [here](./Project-marking.pdf).

<hr>

### Important dates

* Week 7
   * An [Excel sheet](https://docs.google.com/spreadsheets/d/1iFfQNbrxsHOUTYbWlprO49NRwFXYtiJJekvHf6CbXUI/edit?usp=sharing) for sharing interests on project topics released
   * A [Google form](https://docs.google.com/forms/d/e/1FAIpQLSd2ETyh0fNqJSMYrDPWmwl5dsecP_SkMh0YMCTHVFSfF6fpaA/viewform?usp=sf_link) for collecting information about group formation released
* Week 9
   * A Q\&A session at the end of lecture
* Week 10
   * Groups formed
   * Groups are given access to their project repositories on GitHub
   * Each group submits a **project proposal** in a PROPOSAL.md file in their GitHub repository
* Week 11
   * All project proposals approved
* **Solution deadline:** 06/05/2025, 5 pm (both GitHub and Moodle)
* Feedback and provisional marks: in 6 weeks (tentative)

<hr>
