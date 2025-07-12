**Chunk Title:** Eddy's Career Journey & Background  
**Start - End:** 00:00:00 - 00:02:14  
**Transcript:**  
[00:00:00] Host: Let’s get started. This week, we’ll discuss Digital Data Warehousing and FinOps. We have a special guest today, Eddy Zulkifly, a staff data engineer at Kinaxis. Eddy, did I pronounce it correctly?  
[01:35] Guest: Yes, it’s Eddy Zulkifly and Kinaxis.  
[01:40] Host: Okay, great. Eddy is building a robust data platform at Kinaxis and has over 10 years of experience in the data field. He’s also a mentor and teaching assistant. Previously, he worked as a senior data scientist at Home Depot, specializing in e-commerce and supply chain analytics. Welcome to the show, Eddy!  
[01:57] Guest: Thank you, Alexey. It’s a pleasure to be here. I’ll briefly introduce myself. Alexey covered a bit about my experience, but I’ll dive in further.  
[02:03] Guest: Before we go into the main topic of Digital Data Warehousing and FinOps, let’s start with your background. Can you tell us about your career journey so far?  
[02:14] Guest: For sure. My journey into data wasn’t exactly linear. Initially, I was supposed to pursue chemical engineering, but midway through considering my options, I was drawn to industrial engineering, particularly the psychology side—designing systems for people, which is essentially user experience. Although I didn’t know it at the time, that focus on building systems for people shaped how I approach data today.  
[02:14] Guest: My first job was in technology, deep inside supply chains, working in distribution centers. I focused on optimizing space in warehouses, using Excel macros to figure out how many containers would arrive next week or next month. This was tied to staffing demands.

---

**Chunk Title:** Transitioning from Data Science to Data Engineering  
**Start - End:** 00:06:20 - 00:07:48  
**Transcript:**  
[06:20] Guest: I think of FinOps as process optimization, similar to my early work in a physical warehouse but now applied in a digital warehousing context. A lot of the skills I learned early on still apply today, from Excel all the way to Python, Git, and the command line.  
[06:33] Guest: What I also noticed is that you’re currently a data engineer, but you were previously working in data science. How did the switch happen?  
[06:33] Host: I’d like to clarify that my title is senior data engineer, but I moved from a business analyst role to more of a data engineering role. Working as a business analyst, my focus was on building dashboards. That’s where I first learned Tableau Public and attended events like Makeover Monday, where you were given a dataset and had to create narratives and charts.  
[07:48] Guest: Absolutely, I think a data analyst background makes the transition to data engineering easier. The foundational knowledge is already there, especially in understanding how data flows and what the needs of the business are. The skills in analysis and creating reports, dashboards, and interpreting data directly translate.

---

**Chunk Title:** Learning Tools: Command Line & Docker  
**Start - End:** 00:08:00 - 00:09:44  
**Transcript:**  
[08:00] Guest: It’s great to hear that. A lot of people are curious about this shift. One challenge often mentioned in data engineering is working with the command line, Docker, and Terraform. How was your experience transitioning to these tools?  
[08:06] Host: Yes, I agree. When I started learning Docker and Terraform, it felt overwhelming because as a data person, we’re often used to working with UI tools. I came from an Alteryx background, which has a low-code approach. Moving to the command line was a big change. But once I started getting used to it, it all clicked.  
[08:17] Guest: Now, working in data engineering, Docker and Terraform are critical tools for building infrastructure and managing environments. I realized that once you get comfortable with one language or tool, it’s easier to pick up others. It’s about building a mindset where you can apply the concepts across different technologies. It’s not as daunting as it seems at first.  
[09:03] Host: It’s great that you brought that up. I can see how that shift would be a challenge. But now that you’re more comfortable with Docker and Terraform, do you think it’s easier to learn new tools?

---

**Chunk Title:** Excel's Role in Warehousing  
**Start - End:** 00:12:31 - 00:14:44  
**Transcript:**  
[12:31] Guest: The most interesting time for me was during the holiday seasons, particularly September and December. That’s when Home Depot sold a lot of Halloween products, like skeletons and dinosaurs. A typical workflow for me during that period involved forecasting based on orders. We had to figure out how many containers of goods were arriving, how they were packaged, and what products they contained.  
[12:56] Guest: Once we figured out the size of the boxes, we had to estimate how much space they would occupy in the warehouse. After that, we calculated how much labor was required to store them. Later, when we needed to send the products to stores, we’d figure out how many people were needed to pick, package, and ship the items. There were many questions to consider, and we used various tools, like Excel macros, to determine the best configuration to store these products.  
[14:39] Host: I had to pick up a lot of skills. That was my introduction to programming. I remember using the "record" function in Excel to create macros by performing actions like selecting cells, then seeing the code. It was a great starting point.  
[14:44] Guest: I even created a website once. I had a database of music bootlegs—concert videos that fans film themselves. I exchanged DVDs of these recordings with people from all over the world. Using Excel, I created a website. It had a macro that published an HTML page and then uploaded it via FTP. At the time, it was mind-blowing to me.

---

**Chunk Title:** Experiences at Home Depot  
**Start - End:** 00:16:43 - 00:20:47  
**Transcript:**  
[16:43] Guest: I just checked the competition, and it was Home Depot Product Search. It was about predicting the relevance of search results on Home Depot’s website. People would type a search, and the task was to predict the relevance of the products that came up. It was one of my first exposures to these kinds of problems.  
[17:22] Host: That sounds like it was related to cosine similarity.  
[17:27] Guest: I didn’t realize that Home Depot had physical stores. I thought it was just an online store because I participated in that competition and knew the name. They have a big team, and I thought it was just an online store for screws. But it turns out they have a lot of physical stores too.  
[17:51] Host: Yes, Home Depot has a lot of stores in Canada, the US, and Mexico. It’s a pretty big operation.  
[17:59] Guest: It’s a huge operation, and you need a huge warehouse to manage all that inventory. You were working on optimizing those warehouses, right?  
[18:12] Host: Yes, I was involved in the backend processes. We had distribution centers for larger products, which were shipped to stores. On the merchandising side, we had planograms, which are configurations that show where products should be placed in the store. There’s a whole science behind this.  
[19:40] Guest: Is this part of merchandising an art or a science?  
[19:46] Host: It’s called assortment planning. Each store has a slightly different plan based on the region and customer preferences. We would tweak the planograms based on sales data to match local buying habits.  
[20:08] Guest: Did you ever work in a physical warehouse, like actually handling the products?  
[20:14] Host: Yes, my first job at Home Depot was at a distribution center. I was involved in loading products onto racks. Later, I worked more on the corporate side, focusing on planograms and data analytics.  

---

**Chunk Title:** Warehousing: Physical vs Digital  
**Start - End:** 00:19:17 - 00:26:27  
**Transcript:**  
[19:17] Eddy: Is this part of merchandising an art or a science?  
[19:40] Alexey: It’s called assortment planning. Each store has a slightly different plan based on the region and customer preferences. We would tweak the planograms based on sales data to match local buying habits.  
[19:46] Eddy: Did you ever work in a physical warehouse, like actually handling the products?  
[20:08] Alexey: Yes, my first job at Home Depot was at a distribution center. I was involved in loading products onto racks. Later, I worked more on the corporate side, focusing on planograms and data analytics.  
[20:14] Eddy: Did you use robots from Amazon Robotics?  
[20:40] Alexey: I’m not sure. At the time, I don’t think we were using Amazon Robotics, but it’s possible we were testing some automation.  
[20:47] Eddy: Amazon Robotics, previously Kiva Systems, uses robots to move products around in warehouses. I worked at Kiva as a Java developer. I think Home Depot was one of their customers, but maybe that didn’t go anywhere after Amazon acquired them.  
[21:04] Alexey: That’s interesting. I worked with the Canadian branch of Home Depot, so the situation may have been a bit different here.  
[21:18] Eddy: You worked later in your career to optimize distribution centers, which are essentially warehouses, right?  
[21:43] Alexey: Yes, that’s correct. It’s very much like working in a warehouse.  
[21:49] Eddy: We also wanted to talk about digital warehousing. What exactly is that, and how does it work? Let's start with physical warehouses first. A warehouse is a space with tracks where products are stored. There are processes for people, robots, or machines to move the products and deliver them to the appropriate location. For example, getting products from racks and loading them into cars for distribution. That's a physical warehouse.  
[21:57] Alexey: In this case, I'd like to thank you for that context. I’ve worked at Home Depot for a long time, and I recall working in a physical warehouse. But when I became a senior data engineer, I realized I was still working in a warehouse—just a digital one. In this case, think of moving data like moving products. You're ingesting data, similar to trucks delivering goods into a warehouse. We use Google Cloud and BigQuery to store the data. Inside the warehouse, you have internal processes. We used orchestrators to run SQL queries, which helped transform and process the data. Once the data is ready, it is sent to a consumption layer, like a BI tool, such as Tableau or Looker.  
[22:36] Eddy: Data engineers build pipelines or write Python scripts to move data from the source into the warehouse. Inside the warehouse, we use transformation tools like data build tools to clean and organize the data. The data is then put into tables or racks to be easily accessible. Once it’s organized, you have a service account or process to send the data to tools like Looker or PowerBI. Finally, dashboards are created for the end users to generate insights. The process of working in a digital warehouse is quite similar to a physical one.  
[23:38] Eddy: I just realized that the term "data warehouse" includes the word "warehouse," which is really a distribution center. The goal of a physical warehouse is to receive products in large trucks and then store them in racks. The products are organized in a way that makes them easy to access. Once the products are organized, they can be moved to smaller trucks for delivery to stores. In the same way, data in a digital warehouse is organized to make it easy to access.  
[24:34] Alexey: One major difference between digital and physical warehouses is the speed of changes. In my first role in process improvement, we focused on understanding processes and managing change. In a physical warehouse, changes often take a long time because you need approval from various teams. For example, if you need more space or racks, it can cost millions of dollars and require construction.  
[25:32] Eddy: In a digital warehouse, changes are much quicker. If you need a new data requirement, you can create a new table and optimize it immediately. The feedback loop is much faster, too. You can create a model, send it out, receive feedback, and make adjustments quickly. Another difference is that, in a physical warehouse, you might rely on internal systems or human observation to track products. In a digital warehouse, since everything is in the cloud, you need monitoring systems and tests to ensure everything is working properly.

---

**Chunk Title:** The Role of Data Engineering  
**Start - End:** 00:27:50 - 00:29:56  
**Transcript:**  
[27:50] Alexey: My day-to-day work involves both business and technical tasks. One key thing I've learned is the importance of understanding business requirements. In the analytics space, we focus on building metric trees to understand which metrics the business cares about and what factors affect them. We recently worked on a metric tree for the FinOps team to understand the cost factors inside the data warehouse and cloud platform.  
[27:57] Eddy: Building a metric tree helps everyone gain a better understanding of data and how to build datasets that answer important business questions. Another important task is using a data spec, which I learned from Zack Wilson’s course. Business requirements can be vague, so it's important to translate them into clear technical data requirements. This way, you can ensure the data pipeline meets business needs, define metric definitions, set pipeline frequencies, and identify any assumptions or unknowns.  
[28:21] Eddy: Once you have everything outlined, you can communicate with the business and get alignment. This alignment is crucial because it gives everyone a clear understanding of the direction. I believe this step is one of the most important tasks for data engineers. Once the requirements are clear, it's easier to move forward with building the necessary systems.  
[29:16] Eddy: Another part of my role involves standing up a digital warehouse by setting up our stack. We use cloud platforms and a mix of open-source tools to maximize business value while ensuring the system is easy to maintain for engineers and analysts within the team.

---

**Chunk Title:** Introduction to FinOps  
**Start - End:** 00:31:35 - 00:41:55  
**Transcript:**  
[31:35] Eddy: Can you tell us more about what exactly you mean by optimizing costs? Let's say I work at, I don't know, something similar to Home Depot—just as an example, a large chain that needs a solution for supply chain planning. We have warehouses, stores, and clients, and we need to optimize that. Now, if we wanted to use something like InkaNexis, how do we optimize costs? What does it mean in this particular case?  
[31:40] Alexey: In this case, we enter the realm of Software as a Service (SaaS). Many solutions we build eventually run on servers or in data centers. Every component requires some kind of virtual machine with specific requirements for RAM and storage. Additionally, when storing data in different regions, especially with global clients, we must comply with local data regulations, so data has to be stored in certain places. There's also the issue of securing and backing up the data, which involves internal processes to ensure the customers' data is safe and protected. The goal is to figure out how to do all this in the most cost-efficient way possible.  
[32:06] Eddy: So, the cost optimization is for you, not for the clients?  
[33:33] Alexey: Yes, in this case, it's about optimizing costs on our side as well.  
[33:41] Eddy: I see. My impression was that you offer a bunch of solutions, and then the customer comes and says, "I want your solution." Then you work with them to figure out the best setup for them.  
[33:51] Alexey: We also do that. When a client signs on, there’s an engagement process where we work with the customer to integrate their systems into our platform. We have a team dedicated to maintaining that and ensuring it runs smoothly.  
[33:57] Eddy: This is probably related to FinOps, right? Optimizing costs?  
[34:10] Alexey: Yes, exactly. In a previous company I worked for, the procurement department was responsible for tools that cost money. For example, if I needed something like Klaviyo or Google Workspace, I would go through them to figure out how to purchase these tools. For AWS, we could negotiate special contracts based on how many instances we wanted in advance, which made them cheaper. This is a form of cost optimization. So, when you deal with AWS, you negotiate based on the amount of money you want to spend and the services you need.  
[34:15] Eddy: Yes, exactly. Our team uses the data we generate to understand the costs and usage of our systems. From there, we figure out how to negotiate with cloud vendors to optimize the costs. Typically, we have an idea of how long our servers will run, and we can negotiate discounts for longer-term commitments.  
[35:28] Eddy: So, it’s pretty similar to what I just described?  
[35:57] Alexey: Yes, exactly. Reservation instances are part of the FinOps process as well.

---

**Chunk Title:** Cost Management and Tools  
**Start - End:** 00:41:55 - 00:46:17  
**Transcript:**  
[41:55] Eddy: Can you tell us more about the problems you solve and what kind of solutions you use to address them?  
[36:11] Alexey: When addressing cost and usage, it’s important to understand both the business requirements and what’s expected in the coming months. For example, forecasting usage helps in understanding costs. Virtual machines generate significant costs, so knowing when and how long to run them is crucial. Configuring virtual machines requires knowing the required RAM and storage, as these are cost factors. Additionally, depending on the operating system—whether Linux or Windows—there may be licensing considerations, and cloud platforms may offer discounts or bundle licenses.  
[36:18] Eddy: The cloud pricing model is complex, with different factors like RAM, storage, and licensing that contribute to costs. To manage this, we run models across different cloud providers to compare which offers the best value.  
[37:53] Eddy: So, you evaluate multiple clouds, right? You know the requirements for virtual machines, and then you compare the deals you can get from Google Cloud, AWS, Azure, and so on. Based on that, you can make a decision on which provider to go with?  
[38:32] Alexey: That’s correct.  
[38:53] Eddy: Interesting. When it comes to tools or solutions to these problems, it sounds a lot like demand forecasting to me.  
[39:03] Alexey: Right. There’s demand forecasting, inventory planning, and other features. We also do a form of what-if analysis, similar to Excel, where you input different variables to see the potential outcomes. We use a more advanced version internally that looks at parameters and adjusts them to optimize revenue or profits.  
[39:09] Eddy: So, inventory planning and demand forecasting in the cloud is similar to physical operations, right? It sounds like we’re talking about distribution centers and similar physical optimizations.  
[39:47] Alexey: Yes, absolutely. My experience from that part of my career has been very helpful here.

---

**Chunk Title:** FinOps and Data Engineering Insights  
**Start - End:** 00:44:41 - 00:49:37  
**Transcript:**  
[44:41] Eddy - line: While you were explaining, I realized that even though FinOps isn’t directly related to DevOps, there are similarities. In DevOps, the focus is not just on tools but also on processes—ensuring software is reliable, testable, and delivered efficiently. FinOps appears to have a similar focus on processes, streamlining cost optimization efforts, and leveraging tools to achieve this. Would you say that’s accurate?  
[46:17] Alexey - line: Yes, exactly. You hit the nail on the head. The processes in FinOps mirror some of the DataOps practices as well. For example, using CI/CD pipelines to validate datasets or check how new data impacts downstream dashboards employs similar methodologies.  
[47:09] Eddy - line: When I previously asked about your role as a Staff Data Engineer, you mentioned business and technical aspects. Could we revisit that discussion now that we’ve talked about FinOps?  
[47:41] Alexey - line: Sure! As a Staff Data Engineer, my role is multifaceted. On the technical side, I work on deploying pipelines, fixing bugs, and maintaining data quality through DataOps processes. On the strategic side, I focus on defining business metrics like unit economics, which help optimize costs and performance. This dual focus allows us to build a robust data platform for FinOps.  
[48:01] Eddy - line: So, if I understand correctly, you’re building a data platform specifically for FinOps, enabling cost optimization and better decision-making?  
[48:51] Alexey - line: That’s right. We manage data from cloud platforms, apply business logic and metric definitions, and generate unit economics to evaluate cloud performance.  
[49:04] Eddy - line: Does Kinaxis have a dedicated FinOps team? How do data engineers collaborate with them?  
[49:29] Alexey - line: We work closely with various business users, including engineers, product owners, and infrastructure teams. These stakeholders play a role in managing and optimizing cloud spending, ensuring efficient collaboration across the board.

---

**Chunk Title:** Educational Journey and Challenges  
**Start - End:** 00:49:37 - 00:55:14  
**Transcript:**  
[49:37] Eddy - header: Eddy’s master’s degree and closing thoughts - line: Alexey  
[49:37] Eddy - line: I'm almost done with my master's, hopefully by the end of this year. I'm doing a program called the Master’s in Analytics from Georgia Tech. So far, in terms of my learnings, it’s been pretty cool because it’s more of an applied analytics degree. Since I work in data quite a bit, you get exposed to different approaches in analytics.  
[50:25] Eddy - line: We explore descriptive, prescriptive, and predictive analytics, figuring out how to implement data to solve business problems in this context.  
[50:32] Eddy - line: Since you're based in Toronto and this is Georgia Tech, I assume it's...  
[50:32] Alexey - line: '...in Atlanta, in the United States. It’s interesting because my organization, Home Depot, is also based in Atlanta. While taking the program, I met a lot of Home Depot colleagues from the U.S. who were also enrolled.'  
[51:11] Eddy - line: That’s how I learned about cool open-source technologies like dbt (data build tool), picked up Python, explored the cloud, and a range of other tools. These experiences helped me transition from being more of a business user to a software and data engineer.  
[51:11] Eddy - line: When I was doing my master’s in business intelligence, it was almost 10 years ago. This year marks the 10th anniversary since I graduated.  
[51:45] Alexey - line: Back then, I was working as a freelancer. It was more of a part-time job. I’d take a contract, complete it, and then move on to the next one, spending around 20 hours per week. You, however, work full-time as a staff data engineer. It must be challenging to manage both at the same time, right?  
[51:45] Alexey - line: Yeah, exactly. One of the things that interested me about the Georgia Tech program was its analytical rigor. Based on reviews, they expect you to pick up on the math and learn Python as required.  
[52:25] Eddy - line: In one of the courses, we had to learn D3, a JavaScript library, which was pretty intense. I don’t think many people use D3 now, but it was a great opportunity to learn something new and apply it.  
[52:25] Eddy - line: Pursuing my master’s while working full-time has been about managing my time and responsibilities carefully. I try not to overwork myself, usually taking one course per semester. I also choose topics related to my job so I can apply what I’m learning directly at work.  
[52:25] Eddy - line: This approach has added value because I can immediately apply my university learnings to my job in analytics. It helps reinforce my understanding and makes the knowledge more practical.  
[53:55] Eddy - line: This is something we talked about. You also run—you’re into running and were running marathons. This year, you’re slowly returning to that. Preparing for a marathon takes a lot of time, and then you’re also doing a master’s, working, and probably want to do something else apart from these things, right?  
[54:16] Alexey - line: Absolutely. The secret is to take one course per semester.  

---

**Chunk Title:** Learning and Community Engagement  
**Start - End:** 00:55:14 - 00:58:47  
**Transcript:**  
[55:14] Eddy - line: When I was preparing for a marathon, I think I took a lighter course. One thing I’ll say about the analytics community at Georgia Tech is that it’s a really great online community. They have a Google Sheet where they collect course reviews and calculate the estimated workload for each course.  
[54:22] Eddy - line: My rule is to do the math—if I want to double up, my max is 20 hours. If a course is around 10, 8, or 12 hours, I’ll likely just take one at a time. It’s interesting that people in the community spend time building tools like this to help manage workloads. I found that really helpful as well.  
[55:14] Eddy - line: That makes sense. The program is about analytics, so there should be analytical ways to support decision-making in this case, right?  
[55:23] Alexey - line: Exactly!  
[55:31] Eddy - line: I can see your cat.  
[55:37] Alexey - line: Maybe one last question, and then we’ll call it a day. You mentioned you’re either recently preparing for or have already earned a certificate in some field. You seem to have quite a few certificates. Are you actively investing your time in extra learning, in addition to your master’s? How do you manage that, given we only have 24 hours in a day?  
[56:05] Alexey - line: Exactly, and I think this applies to everyone. Nowadays, when I talk to folks interested in breaking into data, I encourage them to focus on certifications that align with their interests and passions.  
[56:05] Eddy - line: As a data engineer, you develop the skill of learning as you go. This has helped me become more efficient in learning and strategic in acquiring the knowledge I need to get the job done.  
[56:05] Eddy - line: Pursuing my master’s and picking up certificates has taught me to build habits around learning. Finding your community is essential. Toronto, for example, is a big tech hub for data and analytics. Attending meetups and learning from others in the city has helped me build relationships and understand market trends.  

---

**Chunk Title:** Mentorship and Giving Back  
**Start - End:** 00:58:47 - 00:59:54  
**Transcript:**  
[57:01] Eddy - line: Another thing I’ve found valuable is applying what I learn. At my organization, there’s openness to building and applying skills from school. Shoutout to my team at Home Depot for creating a great environment for learning and growth.  
[57:01] Eddy - line: Lastly, having an accountability partner helps. I have monthly calls with a friend where we discuss our goals. They’re not even in data, but it helps to have someone to keep you on track.  
[58:47] Eddy - line: Finally, I believe in giving back to the community. Platforms like Data Talks Club are great because they teach others for free. Last year, I joined ADP List as a mentor to share what I’ve learned.  
[58:47] Eddy - line: I wish someone had guided me 10 years ago when I started my career as a data engineer. Engineers are often too busy fixing things to mentor others. I hope to inspire others to pursue this path and succeed.  
[59:32] Eddy - line: Thank you, Eddy. It was amazing talking to you. Thanks a lot for joining us today and sharing your experience.  
[59:32] Alexey - line: I took a lot of notes—it was a very productive discussion. Now I know what FOPS is; I had no idea this even existed. Thanks for taking the time. It was truly amazing.  
[59:32] Alexey - line: Thank you so much, Alexey. It was great to be part of the talk today.