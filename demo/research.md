# 🏆 Chosen PDF Library: WeasyPrint


## At a Glance — Usage & Community Stats

### WeasyPrint
----------
- It has 7.6k [Github](https://github.com/Kozea/WeasyPrint) stars which implies that it is strong and mature.
- It is also regularly [maintained](https://github.com/Kozea/WeasyPrint)
- At the moment of writing this it has about 150 open issues on Github and it is regularly maintained as well judging from the last commits
- It's pip installation statistics, [last month download](https://pypistats.org/packages/weasyprint) of ~ 4 million and a last week download of ~ 1 million
- It has a [BSD 3-Clause License, license](https://github.com/Kozea/WeasyPrint/blob/main/LICENSE) which means it can be used in an open-source app as well.


### fpdf2
-----
- It has about 1.2k Github stars of [fpdf2](https://github.com/py-pdf/fpdf2) so it is growing
- At the moment of writing this it has about 50 open [issues](https://github.com/py-pdf/fpdf2/issues) on Github, and it is regularly maintained as well judging from the last commits
-- It's [pip installation statistics](https://pypistats.org/packages/fpdf2) has a last month download of ~ 1 million and a last week download of ~ 480 thousand
- It has a [LGPL license](https://github.com/py-pdf/fpdf2/blob/master/LICENSE) which means it can be used in an open-source app as well.

### reportlab
-----
- Could not find the open source repo for this, so mean it might be [privately maintained by ReportLab](https://www.reportlab.com/support/), hence we are dependent on ReportLab directly for updates and changes
- It has a high usage and it's [pip installation statistics](https://pypistats.org/packages/reportlab) has a last month download of ~ 6.8 million and a last week download of ~ 1.8 million
- It has a BSD license which means we can use the open-source version for our open-source app

## Developer Experience

### WeasyPrint
- Learning curve is relative, but genrally quite low since it builds layouts using HTML and CSS.
- There is more control in terms of styling a page, because of CSS functionalites
- Good support for images and different fonts even CDN hosted fonts (since it allows for HTML and CSS)
- Allows for reusable templates (HTML, CSS)
- Fit for different document styles

### fpdf2
- Allows for reusable templates
- Seems to be better fit for tabular document styles, e.g invoices, etc
- Styling is also done programmatically

### reportlab
- Learning curve is medium, as it not requires Python but also some coordinate geometry for positioning objects (text, images, etc) in PDFs
- Allows for templating as well (with python functions) but might be more complicated as it coordinate geometry is required

## Evaluation
After evaluating the highlighted libraries above: including WeasyPrint, ReportLab, and FPDF2 for PDF generation libraries — I selected WeasyPrint as the most suitable tool for this project.


## Why WeasyPrint Wins

### HTML & CSS Styling
Layouts, spacing, fonts, and positioning are all handled through familiar CSS, removing the need to calculate (x, y) coordinates or use low-level drawing APIs.

### Modern Design Capabilities
It relies on web standards, WeasyPrint supports modern fonts, flexbox/grid layouts, circular images, and overlays — ideal for polished, presentation-style PDFs.

### Clean and Maintainable Codebase
It is easy to build a reusable base HTML/CSS template and update it for different episodes. This keeps the layout consistent while simplifying future content updates.


After evaluating popular open-source PDF generation libraries — including WeasyPrint, ReportLab, and FPDF2 — I selected WeasyPrint as the most suitable tool for this project.
