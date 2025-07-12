After analyzing the transcript, I've identified the main themes and grouped the consecutive segments into thematic chunks. Here are the results:

**Chunk 1: Introduction and Eddy's Background**
**Start - End:** 0:00 - 2:14
**Transcript:**
[0:00] Alexey: Let’s get started. This week, we’ll discuss Digital Data Warehousing and FinOps. We have a special guest today, Eddy Zulkifly, a staff data engineer at Kinaxis. Eddy, did I pronounce it correctly?
[1:35] Eddy: Yes, it’s Eddy Zulkifly and Kinaxis.
[1:40] Alexey: Okay, great. Eddy is building a robust data platform at Kinaxis and has over 10 years of experience in the data field. He’s also a mentor and teaching assistant. Previously, he worked as a senior data scientist at Home Depot, specializing in e-commerce and supply chain analytics. Welcome to the show, Eddy!
[1:57] Eddy: Thank you, Alexey. It’s a pleasure to be here. I’ll briefly introduce myself. Alexey covered a bit about my experience, but I’ll dive in further.
[2:03] Alexey: Before we go into the main topic of Digital Data Warehousing and FinOps, let’s start with your background. Can you tell us about your career journey so far?
[2:14] Eddy: For sure. My journey into data wasn’t exactly linear. Initially, I was supposed to pursue chemical engineering, but midway through considering my options, I was drawn to industrial engineering, particularly the psychology side—designing systems for people, which is essentially user experience. Although I didn’t know it at the time, that focus on building systems for people shaped how I approach data today.

**Chunk 2: Career Journey and Transition to Data Engineering**
**Start - End:** 2:14 - 6:33
**Transcript:**
[2:14] Eddy: My first job was in technology, deep inside supply chains, working in distribution centers. I focused on optimizing space in warehouses, using Excel macros to figure out how many containers would arrive next week or next month. This was tied to staffing demands.
[2:14] Eddy: I moved through several roles—supply chain, e-commerce merchandising, and business intelligence. However, it wasn’t always about Python or SQL. My first role was in a distribution center, where I had to move out of the city and come into the office. While it wasn’t the typical data analyst role, it was a great learning experience. Looking back, it provided a solid foundation for analytics. Excel remains the universal language for business. No matter how nice your dashboard is, the main question from stakeholders will always be, “Can I have this in an Excel file?”
[2:14] Eddy: My journey into data was driven by curiosity. I didn’t have access to Tableau at first, so I used Tableau Public, learning from YouTube and different creators. Eventually, my organization gained access to Alteryx, which I loved because it allowed quick analysis with a low-code approach. I got certified in Alteryx.
[2:14] Eddy: At the same time, I was pursuing a master’s in analytics, which led me to discover Python and build modern data stacks with an ELT approach. Now, I work as a staff data engineer at Kinaxis, focusing on building data sets and dashboards for our FinOps team to optimize cloud spend and ensure the platform runs efficiently.
[6:20] Alexey: I’d like to clarify that my title is senior data engineer, but I moved from a business analyst role to more of a data engineering role. Working as a business analyst, my focus was on building dashboards. That’s where I first learned Tableau Public and attended events like Makeover Monday, where you were given a dataset and had to create narratives and charts.
[6:33] Eddy: Moving into data engineering, I realized my true passion lay in the technical side. I had a great manager who told me that whenever I discussed Tableau or dashboards, my eyes would light up. That’s when I realized I enjoyed working on the backend side of things. That’s why I decided to pivot my career to data engineering.

**Chunk 3: Transition to Data Engineering and Learning New Tools**
**Start - End:** 6:33 - 10:00
**Transcript:**
[6:33] Eddy: This is a common question in our data engineering course. Our course is designed for people coming from a software engineering background, but we often get asked, “I’m a data analyst—can I transition to data engineering?” Does the background of an analyst make this transition easier?
[7:48] Alexey: Absolutely, I think a data analyst background makes the transition to data engineering easier. The foundational knowledge is already there, especially in understanding how data flows and what the needs of the business are. The skills in analysis and creating reports, dashboards, and interpreting data directly translate. The main difference will be shifting from more front-end roles, like working with Tableau or Power BI, to focusing on back-end processes, such as data pipelines, databases, and automating workflows.
[8:00] Eddy: It’s great to hear that. A lot of people are curious about this shift. One challenge often mentioned in data engineering is working with the command line, Docker, and Terraform. How was your experience transitioning to these tools?
[8:06] Alexey: Yes, I agree. When I started learning Docker and Terraform, it felt overwhelming because as a data person, we’re often used to working with UI tools. I came from an Alteryx background, which has a low-code approach. Moving to the command line was a big change. But once I started getting used to it, it all clicked.
[8:17] Eddy: Now, working in data engineering, Docker and Terraform are critical tools for building infrastructure and managing environments. I realized that once you get comfortable with one language or tool, it’s easier to pick up others. It’s about building a mindset where you can apply the concepts across different technologies. It’s not as daunting as it seems at first.
[8:42] Alexey: I have a similar experience. When I first encountered Docker, I was introduced to development containers. I had some colleagues in Toronto who were explaining development containers using a JSON file. When I tried explaining it to data folks, they found it difficult to grasp because it wasn’t a standard approach like other tools we use. It’s not really DevOps; it’s more about configuring a file. Working with Docker and development containers requires some getting used to.
[9:03] Eddy: It’s great that you brought that up. I can see how that shift would be a challenge. But now that you’re more comfortable with Docker and Terraform, do you think it’s easier to learn new tools?
[9:28] Alexey: Definitely. The key is the mindset of understanding the fundamentals and getting comfortable with one tool first. From there, it’s much easier to pick up new tools because the concepts tend to overlap. Whether you’re working with Python, Terraform, or any other tool, they all serve a similar purpose—they help automate processes and manage infrastructure efficiently. The more you work with these tools, the more it feels natural.

**Chunk 4: Experience at Home Depot and Working with Excel**
**Start - End:** 10:00 - 17:27
**Transcript:**
[10:00] Eddy: If all you did before was business analytics, and now you suddenly need to work with Docker, everything related to this feels overwhelming. It might seem like devops, something complicated. But once you dive into it, you realize it's not devops; it's something different, something you can learn.
[10:49] Alexey: You worked at Home Depot, and for me, Home Depot was notable for hosting a Kaggle competition. I think it was one of the first Kaggle competitions I participated in. Was it about forecasting? I can’t quite recall, but I remember it being an awesome experience. Can you tell us more about your work there? Were you still focused on physical warehouses, or were you already transitioning to digital warehouses?
[11:13] Alexey: My time at Home Depot was a discovery process where I explored different aspects of my degree. Part of my degree focused on logistics and optimization, so I had periods where I worked in the warehouse. To give context, think of the equivalent of Home Depot in Europe as something like an Obi or Bauhaus. In Southeast Asia, where I'm from, it could be something like Mr. DIY or Daiso. These are stores that sell products like screws, hammers, and outdoor furniture.
[11:54] Eddy: The most interesting time for me was during the holiday seasons, particularly September and December. That’s when Home Depot sold a lot of Halloween products, like skeletons and dinosaurs. A typical workflow for me during that period involved forecasting based on orders. We had to figure out how many containers of goods were arriving, how they were packaged, and what products they contained.
[12:31] Eddy: Once we figured out the size of the boxes, we had to estimate how much space they would occupy in the warehouse. After that, we calculated how much labor was required to store them. Later, when we needed to send the products to stores, we’d figure out how many people were needed to pick, package, and ship the items. There were many questions to consider, and we used various tools, like Excel macros, to determine the best configuration to store these products.
[12:56] Eddy: At the time, we also did something called preload optimization, which involved engineers and warehouse staff figuring out how to maximize trailer space when shipping products to stores. We manually solved problems like the knapsack problem using Excel, which was quite manual. Looking back, I realize that today software from companies like Kanexis automates a lot of this work, making it much easier.
[13:38] Eddy: Interesting! You must be an expert with Excel macros.
[14:39] Alexey: I had to pick up a lot of skills. That was my introduction to programming. I remember using the "record" function in Excel to create macros by performing actions like selecting cells, then seeing the code. It was a great starting point. Over time, I learned how to define objects and refine the macros to make them more efficient.
[14:44] Eddy: I even created a website once. I had a database of music bootlegs—concert videos that fans film themselves. I exchanged DVDs of these recordings with people from all over the world. Using Excel, I created a website. It had a macro that published an HTML page and then uploaded it via FTP. At the time, it was mind-blowing to me.
[15:10] Alexey: That’s interesting! I have a similar story, but we can talk about that later.
[16:06] Eddy: Excel is surprisingly powerful. There’s so much you can do with it. You could even play games like Space Invaders or Doom in Excel if you wanted.
[16:13] Alexey: I’ve seen people create art in Excel, and there are even Excel competitions. In those competitions, participants have to create complex models to solve specific problems. If you can do it faster, there are people willing to pay for those skills.
[16:26] Eddy: I just checked the competition, and it was Home Depot Product Search. It was about predicting the relevance of search results on Home Depot’s website. People would type a search, and the task was to predict the relevance of the products that came up. It was one of my first exposures to these kinds of problems.
[16:43] Alexey: That sounds like it was related to cosine similarity.
[17:22] Eddy: I didn’t realize that Home Depot had physical stores. I thought it was just an online store because I participated in that competition and knew the name. They have a big team, and I thought it was just an online store for screws. But it turns out they have a lot of physical stores too.
[17:27] Alexey: Yes, Home Depot has a lot of stores in Canada, the US, and Mexico. It’s a pretty big operation.

**Chunk 5: Working in Physical Warehouses and Merchandising**
**Start - End:** 17:27 - 20:47
**Transcript:**
[17:27] Eddy: It’s a huge operation, and you need a huge warehouse to manage all that inventory. You were working on optimizing those warehouses, right?
[17:51] Alexey: Yes, I was involved in the backend processes. We had distribution centers for larger products, which were shipped to stores. On the merchandising side, we had planograms, which are configurations that show where products should be placed in the store. There’s a whole science behind this.
[18:12] Eddy: We had heat maps that showed where sales were coming from, and a dedicated team worked on optimizing store layouts to maximize revenue. Certain products were placed closer to the exits, while others were at the back to get customers to walk around more. It’s similar to how grocery stores place milk at the back to get people to walk through the aisles.
[18:29] Eddy: Did you use Excel for that as well?
[18:47] Alexey: For merchandising, the team used tools like Alteryx and Tableau. We also had in-house software that mapped sales data to store layouts and planograms.
[19:17] Eddy: Is this part of merchandising an art or a science?
[19:40] Alexey: It’s called assortment planning. Each store has a slightly different plan based on the region and customer preferences. We would tweak the planograms based on sales data to match local buying habits.
[19:46] Eddy: Did you ever work in a physical warehouse, like actually handling the products?
[20:08] Alexey: Yes, my first job at Home Depot was at a distribution center. I was involved in loading products onto racks. Later, I worked more on the corporate side, focusing on planograms and data analytics.
[20:14] Eddy: Did you use robots from Amazon Robotics?
[20:40] Alexey: I’m not sure. At the time, I don’t think we were using Amazon Robotics, but it’s possible we were testing some automation.
[20:47] Eddy: Amazon Robotics, previously Kiva

**Chunk 6: Introduction to FinOps and Digital Warehousing**
**Start - End:** 20:47 - 25:32
**Transcript:**
[20:47] Eddy: We also wanted to talk about digital warehousing. What exactly is that, and how does it work? Let's start with physical warehouses first.
...
[25:32] Eddy: In a digital warehouse, changes are much quicker. If you need a new data requirement, you can create a new table and optimize it immediately.

**Chunk 7: FinOps, FinOps Foundations, Certifications and Resources**
**Start - End:** 25:32 - 41:37
**Transcript:**
[25:32] Eddy: In a digital warehouse, changes are much quicker. If you need a new data requirement, you can create a new table and optimize it immediately.
[31:35] Eddy: Can you tell us more about what exactly you mean by optimizing costs?
[33:51] Alexey: We also do that. When a client signs on, there’s an engagement process where we work with the customer to integrate their systems into our platform.
[33:57] Eddy: This is probably related to FinOps, right? Optimizing costs?
...
[41:37] Alexey: Yes, that's correct. FinOps is all about using the cloud in the most cost-effective way.

**Chunk 8: Learning Strategies, Career Development, and Community**
**Start - End:** 41:37 - 59:54
**Transcript:**
[41:37] Alexey: Yes, that's correct. FinOps is all about using the cloud in the most cost-effective way.
[44:41] Eddy: While you were explaining, I realized that even though FinOps isn’t directly related to DevOps, there are similarities. In DevOps, the focus is not just on tools but also on processes—ensuring software is reliable, testable, and delivered efficiently. FinOps appears to have a similar focus on processes, streamlining cost optimization efforts, and leveraging tools to achieve this. Would you say that’s accurate?
[46:17] Alexey: Yes, exactly. You hit the nail on the head. The processes in FinOps mirror some of the DataOps practices as well. For example, using CI/CD pipelines to validate datasets or check how new data impacts downstream dashboards employs similar methodologies.
[49:37] Eddy: I'm almost done with my master's, hopefully by the end of this year. I'm doing a program called the Master’s in Analytics from Georgia Tech. So far, in terms of my learnings, it’s been pretty cool because it’s more of an applied analytics degree. Since I work in data quite a bit, you get exposed to different approaches in analytics.
...
[59:54] Eddy: --- Links: * [Twitter](https://x.com/eddarief){:target="_blank"} * [Linkedin](https://www.linkedin.com/in/eddyzulkifly/){:target="_blank"} * [Github](https://github.com/eyzyly/eyzyly){:target="_blank"} * [ADPList](https://adplist.org/mentors/eddy-zulkifly){:target="_blank"}