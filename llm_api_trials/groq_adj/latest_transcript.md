# Transcript

Source: [https://datatalks.club/podcast/s20e07-build-strong-career-in-data.html](https://datatalks.club/podcast/s20e07-build-strong-career-in-data.html)

---

The transcripts are edited for clarity, sometimes with AI.
If you notice any incorrect information,
let us know
.
Lavanya’s journey from software engineer to AI researcher
Benchmarking long context language models
Limitations of large context models in real domains
Handling large documents and publishing research in industry
Building a data science career: publications, motivation, and mentorship
Self-learning, hackathons, and networking
Community work and Kaggle projects
Mentorship and open-ended guidance
Building a strong data science portfolio
Lavanya’s journey from software engineer to AI researcher
Alexey
: This week we'll talk about building a strong career in data and we have a special guest today. (
0.0
)
Lavanya
: Yes. (
1:49
)
Alexey
: Great! NLP - tell us more about that. Welcome to the show Lavanya. (
1:56
)
Lavanya
: Thank you, great to meet all of you. (
2:02
)
Alexey
: The questions for today's interview are prepared by Johana. Thanks Yana for your help as always. Before we go into our main topic of building a strong career in data, let's start with your background. Can you tell us about your career journey so far? (
2:02
)
Lavanya
: Happy to share my journey. I'll keep it short for now - we can delve deeper into the interesting parts later. After my undergrad in India in 2016, I worked for a couple of years as a software engineer until I got interested in AI/ML - back then it was more of data science stuff. (
2:22
)
Lavanya
: In 2018 I decided to switch from my software engineering profile to an ML profile, worked for a year in India, then decided to pursue my masters. I came to the US, did my masters from Carnegie Mellon, and now I'm working with JP Morgan in their ML vertical. That's quickly a short overview about my story. (
2:22
)
Alexey
: This is a typical question we ask, but since today's interview is mostly about your career, this question is kind of what the entire interview will be about. How did you actually become interested in machine learning and AI? Was there a specific moment or project that sparked your curiosity? (
3:25
)
Lavanya
: Definitely. The interesting part is that while I was doing my undergrad, I had zero courses in ML or data science - this was back in 2016 when very few specialized people were doing ML. I joined my regular software engineering role with HSBC until I participated in a hackathon. (
3:52
)
Lavanya
: My interest peaked through computer vision - although now I'm deep into NLP, I started my journey in ML with vision models. ImageNet and those kind of things were really hot back in 2016-2017. I participated in a lot of hackathons within my firm where we were given liberty to experiment with new things coming up apart from my regular role. That's definitely a clear moment for me where my interests got piqued over 2-3 years of participating in hackathons. (
3:52
)
Alexey
: Do you remember any of the projects you did there? (
4:47
)
Lavanya
: One project that stood out and actually got converted into a product in the bank was an OCR model. When onboarding customers, the bank received organization charts - complicated flowchart diagrams. Back in 2016 with no fancy LLMs, we built specific models that could parse this structure to find relationships between entities and auto-populate forms showing the organization hierarchy - who reports to who. (
4:55
)
Lavanya
: We used vision models to extract boxes, arrow connections, then BFS/DFS algorithms to find connections, and finally OCR through Google Cloud to extract text within boxes and put it all together. It looks pretty naive today, but was very interesting stuff back then. (
4:55
)
Alexey
: If you ask me now how to solve this problem, I wouldn't know - you could just send it to ChatGPT and it would work. But apart from that I have zero idea how to actually approach that. You were a developer - you probably had even less idea than I do now. (
6:18
)
Lavanya
: It was a software engineering approach - just hacking parts of it together. None of us had an ML background, so we did the most basic things possible. It was a rewarding experience but definitely something that piqued my interests in this field. (
6:53
)
Alexey
: I can imagine - if you have zero knowledge but manage to hack something together in a couple days just by Googling and trying tutorials, and it works in the end, that must be quite rewarding. (
7:18
)
Lavanya
: Exactly. Also back then information was limited so it was easier to evaluate all options and pick one, instead of having 100 possible solutions now. That's another downside today. (
7:30
)
Alexey
: So you're saying before it was easier because there were only handful of articles, and anything you did would come out as innovative because not many people were in this field? (
7:54
)
Lavanya
: Yes. (
8:05
)
Alexey
: These days people wouldn't worry about these things - they would just... (
8:10
)
Lavanya
: I think the first thing people would do now is just ask any of the vision LLMs to do this for you. (
8:17
)
Alexey
: Is it something you do now? You're doing something with LLMs right now, right? (
8:23
)
Lavanya
: As I said, I started with vision but now I'm deeply into NLP, so not doing much on vision side now. (
8:29
)
Currently I'm more into... My team at JP Morgan is specifically focused on benchmarking LLMs. We talk to model providers like OpenAI, Anthropic, Meta - take their models and benchmark them on our internal datasets. Our team is the first entry barrier for any model the bank wants to ingest.
: We benchmark on quality as well as deployment aspects like latency and throughput, then publish developer guidance, blogs, do webinars to share experiences with models and best practices. These models are sensitive to prompting and keep changing with new releases every week. We also negotiate with providers for best rates given the bank's high usage. So I'm heavily into benchmarking LLMs. (
8:43
)
Alexey
: I didn't ask you before - can you tell us more about these projects or is this something you can't discuss because banks take these things very seriously? I know some banks don't allow discussing work details - is that the case? (
9:55
)
Benchmarking long context language models
Lavanya
: We can talk about the published work that we have. That's also a similar domain. Benchmarking can be done on different aspects: traditional NLU, long context, code generation capabilities, math capabilities, and multimodal stuff. (
10:15
)
Lavanya
: My focus for the published work was on the long context capabilities. These models claim they have this huge context window of 128k, but can they really read a 200- or 500-page book and give you the correct answer as they claim? We delved deep into that, so we can definitely talk about it since it's published. (
10:15
)
Alexey
: So, short answer: can they read a 500-page book and give an answer? (
10:50
)
Lavanya
: I'll say yes and no. Yes on easier datasets, and no on real datasets. (
10:58
)
Lavanya
: If you check the published work or public benchmarks, they are really positive and say the models are getting better with time. But there is this concept of artificially simplified tasks, which makes it easier for the model. When you use the same model in the real world, especially in specialized domains like healthcare or finance, you start to see the pitfalls of these models in longer contexts. (
10:58
)
Alexey
: Let's say after we finish recording this podcast, I get the transcript and put it into ChatGPT. If I ask questions based on the transcript, I'm pretty sure it will work fine because it's only one hour. I don't know how many pages that translates to. (
11:40
)
Alexey
: But if I take an Andrew Huberman podcast, like one of the episodes I listened to the other day, it's four hours long-probably the length of a book. The topics go deep and sometimes they talk about the latest research. I'm wondering if the model will be able to actually answer questions correctly there. What do you think? (
11:40
)
Limitations of large context models in real domains
Lavanya
: I have two opinions on that. In our studies, it's hard to say at what range in the context window the models fall, so we split it into less than 32k tokens and greater than 32k tokens. There's a clear dip around that. (
12:36
)
Lavanya
: We decided on 32k tokens because, at least in the bank, all our use cases usually fall within 32k tokens. As you said, maybe this transcript would be way less than that, but maybe the Huberman podcast would be much greater. If it's a shorter input, it's better, and as you push the boundary up to 128k, that's where you start to see these falls in the model capabilities. (
12:36
)
The second thing is the natural language aspect. If you give these models a task to summarize or answer questions, it's easy for them to make up stuff, and it's hard for you to objectively verify whether it's correct or not. The tasks we evaluate the models on are very objective, like auto evals, so there's no subjectivity. For this, you would do something like a ROUGE score, but what we do is very specific, like precision and recall on specific data.
: We can talk more about the task, but that's another aspect. For natural language, the model will definitely give you something and you might think, "Oh, looks nice." (
13:30
)
Alexey
: Because it will start reading. (
14:15
)
Lavanya
: It will still be grounded somewhere in the context but might not be exactly correct. (
14:21
)
Alexey
: Should we, before long context models were available, chunk every page as a document, index it, ask a question, retrieve relevant pages, and then let the LLM summarize these pages to get the answer? (
14:29
)
Handling large documents and publishing research in industry
Lavanya
: Even when we try to use it in our bank, we know these models fail at around the 64k context, even though we are using these fancy 128k models. We do the same thing you said: we chunk it because we know up to this point the models don't fail usually. So we chunk it and then do whatever is the downstream processing. (
14:54
)
Alexey
: And this is related to the paper you published, right? (
15:13
)
Lavanya
: Yes. (
15:21
)
Alexey
: This was the paper we talked about at EMNLP. The name of the paper is "Systematic Evaluation of Long Context LLMs on Financial Concepts." That's exactly the paper, and this is what we're talking about right now, right? (
15:28
)
Lavanya
: Yes, yes, this is exactly it. (
15:34
)
Alexey
: Did you like writing the paper? (
15:42
)
Lavanya
: Yes. In fact, it was my fourth publication. I had some publications back when I was studying, but they were not core NLP-they were all over the place. (
15:42
)
Lavanya
: As I said, I started out with data science and spent some time in visualization, so all my papers were on different topics. This was really rewarding, and it's an ACL conference, which is really reputed in the community. That was really nice. (
15:42
)
Alexey
: ACL is Association for Computational Linguistics? (
16:14
)
Lavanya
: Yes, Association for Computational Linguistics. (
16:20
)
Alexey
: I did my master's also, it was related to NLP. I was working with mathematical formulas in Wikipedia and I remember this abbreviation from those days. It's like one of the top conferences, right? (
16:26
)
Lavanya
: It's really top tier, highly filtered, and attended by really smart and brilliant people, like Andrew Ng and all of these people. You can imagine. (
16:37
)
Alexey
: This is pretty rare, from what I see, to work on a paper while working at a company. Typically, when you work at a company, you are concerned about other things. At most, maybe you write a blog post or give a talk at a conference, but here you wrote a scientific paper for a scientific conference. Is it common for people at JPMorgan Chase to do this? (
16:59
)
Lavanya
: I think so. The vertical I work with is MLCO. We are a group of about 150 people, so we are not tied to a single use case or product. We do work on products, but we also have some creative liberty because we do a lot of new stuff. (
17:28
)
Lavanya
: If anything new comes up, we do publish, although it's in the industry track and we are not allowed to release the data. A lot of us keep coming across new findings. Our teams are divided into NLP, graphs, multimodal, vision, and speech. We have specialized teams for all of these, so there are very active and smart people in each group. If anything new comes up, there's always encouragement to go ahead and publish your work, but it comes in addition to your regular work-it's extra effort. (
17:28
)
Lavanya
: It's always encouraged. (
17:28
)
Alexey
: Extra effort, because I know that writing a paper, especially for a top-tier conference, is not easy. If it comes as extra effort to what you already do-let's say you work 40 hours per week and then you probably need another 40 hours per week for a paper-how did you manage that? (
18:44
)
Lavanya
: This one was definitely something that my manager and I were excited about. When we tried to look up the existing research, we found limited resources, so we knew this would be a unique contribution. (
19:04
)
Lavanya
: It was a lot motivated by knowing the scope, that this is an underexplored area. Maybe for things that are well developed, it needs a lot more thinking through, but for us, as soon as we found something in our experiments, we thought, "Okay, this needs to go out in the public, people are not aware of this." (
19:04
)
Building a data science career: publications, motivation, and mentorship
Alexey
: Since our conversation is about building a strong career in data, even though we talk specifically about your career, I think it's really good for a career to have publications. Most of us just go to work, do some stuff related to work, but at the end, we don't share the results. Even if we do, it's typically not top-tier scientific conferences. (
19:45
)
Alexey
: How can we force ourselves to go this extra mile when it's extra work and it's not simple work, it's complicated work? How do you motivate yourself? You said that there is a unique contribution, but still, I imagine you could be pretty drained after work. Writing papers is super difficult, at least for me. I remember being a master's student and I really hated working on my thesis. That was terrible. (
19:45
)
Lavanya
: When I chose my master's, I had the option of doing a thesis. In undergrad, it was much easier, but I was so bad at it that I didn't want to do a thesis option. It takes so much out of you. (
20:48
)
Lavanya
: But as I said, for me, my manager was a really good motivator. He had more experience in this field and was certain we should definitely put this out. I had moments of doubt, wondering if it was worth it with all the effort, but having guidance or a motivator besides yourself is really helpful. (
20:48
)
Lavanya
: Published work is always valuable, even if it's not accepted to any conference. We were okay with just putting it out on arXiv. Unique contributions always shine, if nothing else. We were certain about at least putting a good quality paper on arXiv. If it gets accepted, great; if not, that's fine. (
20:48
)
Lavanya
: Having the motivation to share it with the community is nice. It's hard, but it's nice. (
20:48
)
Alexey
: I remember I took part in a competition and after the competition, I wrote a summary and, without thinking too much, just uploaded it to arXiv. People keep citing it even now. (
22:10
)
Alexey
: The competition was something like fake news detection. Even now, I recently discovered there's one more citation. It's not overly cited-maybe ten or so-but it was something I didn't really think about. I just had this piece of writing, quickly put it together, generated a PDF, and uploaded it to arXiv. (
22:10
)
Alexey
: You just upload the LaTeX file. I didn't really think much about it, just uploaded it and forgot. Later, I discovered people cited it and was like, wow. (
22:10
)
Lavanya
: That's what I did in my undergrad when I was first getting into this field. People said, "Let's just upload it there," and I thought, who is even going to look at it? There are hundreds and hundreds of publications out there. People want to trust credible sources, so I understand. (
23:04
)
Lavanya
: Even arXiv has really high standards. You can't just upload anything; they check that your paper is well formatted, and you need to be endorsed by existing members. It's still a regulated community. So, even if you upload to arXiv, it's nice. (
23:04
)
Alexey
: Now I remember that I needed to ask a friend to endorse me. There were some categories where you didn't need endorsement. For machine learning, you needed endorsement, but for information retrieval back then, you did not-maybe now things have changed. (
23:43
)
Alexey
: My earlier work, like my thesis, I just uploaded to arXiv because it was kind of related to NLP to some extent. (
23:43
)
Lavanya
: Correct, yes. I uploaded it to computer science, and that required endorsement. (
24:07
)
Alexey
: From what I understand by talking to you now, you were always looking to go the extra mile. When you were working as a software developer, you took part in hackathons. (
24:15
)
Alexey
: Now you write research papers even though it's extra work on top of what you do. You engage in all these extra activities in the space. How do you find motivation to do that, and how has it been beneficial for you? (
24:15
)
Self-learning, hackathons, and networking
Lavanya
: In the hackathon space, going back to that era ten years ago, there were very limited resources. I was pretty active on the web at that time to self-learn. Self-learning was crucial for me because I had just finished my undergrad. (
25:01
)
Lavanya
: I'm not sure if you remember, but we had a conversation back in 2020. I reached out to you randomly on LinkedIn. (
25:01
)
Lavanya
: It was on LinkedIn, yes. We talked for a while. (
25:01
)
I think I was doing some pet project and wanted some help around Docker or AWS Lambda-something in MLOps. You were doing these Zoom Camps back then. There were very limited resources, and I wanted quick, easy, trusted, credible help.
: I went through a tutorial you made, found it interesting, and maybe got stuck at some step because I wanted something different from what you showed. So I reached out to you and asked for help. (
25:39
)
I think I was doing some pet project and wanted some help around Docker or AWS Lambda-something in MLOps. You were doing these Zoom Camps back then. There were very limited resources, and I wanted quick, easy, trusted, credible help.
: That's what I used to do then-quickly seek help and self-learn. It's interesting because I completely forgot about it after we talked. You gave me plenty of resources, I finished the project, thanked you, and forgot about it until this session came up. (
25:39
)
I think I was doing some pet project and wanted some help around Docker or AWS Lambda-something in MLOps. You were doing these Zoom Camps back then. There were very limited resources, and I wanted quick, easy, trusted, credible help.
: I realized, oh my god, I've spoken to Alexey before. I remember talking in detail about MLOps. At that time, we also discussed my roles-I was doing some mentoring, instructing, and developing a course with DataCamp. We talked a lot, all over the place. (
25:39
)
I think I was doing some pet project and wanted some help around Docker or AWS Lambda-something in MLOps. You were doing these Zoom Camps back then. There were very limited resources, and I wanted quick, easy, trusted, credible help.
: That also keeps me motivated, seeing people doing so many new things. You were running Zoom Camp, I was doing my own pet projects. It was great. (
25:39
)
Alexey
: It's cool. I went through our conversation and felt nostalgic about all the things we were doing. We were talking about deploying with Lambda, and back then there was a new feature where you could put Lambda inside a Docker container and serve it. (
27:10
)
Alexey
: I remember making a post about this on LinkedIn and getting a lot of likes-maybe 500 in a minute, though I'm exaggerating. People really liked this stuff back then. (
27:10
)
Alexey
: So, you try to be active, self-learn, and when you're stuck, you reach out to people, make connections, and all this motivates you to do more and give back to the community. (
27:10
)
Lavanya
: Sometimes I feel I'm doing too much-I get too much information from the web. I see something and think, "I want to do this," or "This person is so cool, I want to contribute here." (
28:01
)
Lavanya
: At that time, as I mentioned, I was in a software development profile. My aim was to become a full stack ML engineer, which was a hot term then-to know everything from data processing to data modeling and deployment. (
28:01
)
Lavanya
: I was really interested in that. You were publishing a lot about MLOps, so you were my go-to resource for that. For data processing, I would speak to other people to get pipelines built. (
28:01
)
Lavanya
: I was really inspired by this full stack ML role. (
28:01
)
Alexey
: And today, are you still inspired by this role, or maybe we don't need this role anymore? (
29:00
)
Lavanya
: I don't know if we don't need this role anymore, but I think now I've found my place more in NLP. The roots of software development are long forgotten. (
29:07
)
Lavanya
: I still do some software development, but not so much. I think I've found my calling more in NLP. (
29:07
)
Alexey
: For me, the main idea behind this so-called full stack engineer or data scientist is not being afraid of things you don't know. There are things you need to do, and if not you, then who? (
29:30
)
Alexey
: You can just take responsibility and say, "Okay, I'll try my best to do them." If you're not afraid of doing everything-from backend to frontend-even if you have no clue how it works, but you're willing to figure it out, then you are full stack. (
29:30
)
Alexey
: I think people like that are still needed, especially today, because you have AI tools that can help you. (
29:30
)
Lavanya
: I think early in your career, it's hard but it's nice to get your hands dirty with everything. Then you don't feel like, "Oh, this is something I'm unaware of," or "This doesn't fall into my purview." (
30:14
)
Lavanya
: Now, at least in my role, we use Streamlit a lot. As soon as you develop something, you don't want to wait for engineering or figure out how to pass it on to leadership for feedback. You don't want to be dependent on the engineering team. Streamlit has been a really valuable, quick spin-up tool to share what you've built and gather feedback. (
30:14
)
Alexey
: I remember creating React applications, which with Streamlit would have been much easier. (
30:56
)
Alexey
: Instead of spending an hour figuring out React, you could just use Streamlit. But these days, with all this AI, it doesn't really matter, I guess. (
30:56
)
Alexey
: From what I hear, you were interested in the full stack MLE role, which let you play with different areas. You were able to do things everywhere and then eventually realized the thing you like most is NLP, right? (
30:56
)
Lavanya
: Yes, NLP. I think it's also a consequence of my master's. When I went into my master's, I didn't have the idea that I would go deeper into NLP. (
31:47
)
Lavanya
: But the way my course was structured and the people I interacted with really influenced me. My program was part of LTI at CMU-the Language Technologies Institute-which is highly focused on language and speech technologies. (
31:47
)
Lavanya
: Because of the people I was surrounded with, that influenced me. Going into my master's, I didn't have this thought, but at the end of two years, that's what I realized. (
31:47
)
Alexey
: When we started this conversation, we talked about the hackathon, and then I asked you about what you do. Previously it was vision, now it's NLP. I asked you what you do, and we kind of jumped ten years or so. (
32:38
)
Alexey
: But as we continued talking, you mentioned that you also did mentoring and were an instructor at DataCamp. Can you tell us more about these extra activities you did in addition to your main work or studies? (
32:38
)
Community work and Kaggle projects
Lavanya
: Sure. All of this work was definitely on the side. I think it's just like connecting the dots. Sometimes when I was free, I would attend Zoom camps like yours and others. (
33:24
)
Lavanya
: One thing I attended was a web scraping tutorial. It was always fascinating back then to do web scraping, even with restrictions on websites that would block you. For some reason, I was looking up datasets on Kaggle and realized there were a lot of datasets on the Apple App Store, but not comprehensive ones on Google Play Store. (
33:24
)
At that time, the App Store's UI looked much simpler than Google Play, which had dynamic loading as you scrolled, while Apple was more static. Maybe that's why web scraping was more challenging for Google Play. This was back in 2019, so I'm speaking from memory.
: I tried scraping and put the dataset on Kaggle. At one point, it was the highest voted dataset on Kaggle after the COVID dataset. In 2019, it had around 10-15k upvotes. I just checked before this webinar-currently, it's the tenth highest, but at one point, it was trending as the second highest. (
34:05
)
Stuff like that, maybe it's luck, but also some motivation to do something unique. I published it on Kaggle, wrote a simple notebook with basic EDA and modeling, and then someone from DataCamp reached out to me because it was trending and getting a lot of interest.
: They wanted to onboard it as a paid project for learners. During COVID, self-learning and online learning were at their peak, so I got a lot of attention on that project. I made a guided version with guidelines for learners, then an unguided version. Things just connected one after the other. (
34:58
)
Alexey
: It's really amazing. We were talking about luck, and maybe we can attribute some of this to luck. I remember talking to Eugene Yan, who said something about luck that stuck with me. (
36:26
)
Alexey
: He said, imagine you have a bow and want to shoot an arrow at a target. If you don't shoot any arrows, you won't hit the target. But if you shoot a hundred, maybe at least one will reach the target. (
36:26
)
Alexey
: You need to be shooting, right? (
36:26
)
Lavanya
: I talk about opportunity-there are opportunities standing at your door. You just need to open it and peek out. (
37:11
)
Alexey
: For you, that was attending a tutorial, scraping the Google Play Store, and then it had a snowball effect, bringing more and more opportunities. (
37:19
)
Mentorship and open-ended guidance
Lavanya
: Yes, definitely. DataCamp really highlighted my contribution, and I got more confidence to reach out to more people. (
37:32
)
Lavanya
: I started getting involved in the community aspect, starting with Women in Data Science and small communities with local or regional chapters. I also reached out to more industry players to get more mentorship experience. (
37:32
)
Alexey
: That's really cool. What did you do as a mentor? What kind of things did you help people with? (
38:13
)
Lavanya
: All sorts of things. In one organization, I was an instructor and created my own modules to teach basic data science, EDA, all the way up to modeling. (
38:24
)
Lavanya
: We went from regression to classification, and the last thing we covered was decision trees, random forest, boosting, that kind of stuff. All of this was targeted at introducing candidates to the module, but also with a focus on interviews. (
38:24
)
These organizations offer packages where you collaborate with them and they help with interview prep. So, a lot of mock interviews, teaching, resume refinement, LinkedIn reviews.
: With organizations like Women in Data Science, it was more open mentoring. You could come and talk about whatever stage you were at in your life. (
38:54
)
One time I had a really nice conversation with a college student who was brilliant and smart. There was a lot for me to learn from her too-she had already interned at Google. I was curious about the interview process.
: So, just open mentoring sessions with these communities. It's a two-way exchange. (
39:32
)
Alexey
: Maybe you have an opinion about this. Some might say you got lucky-your dataset trended and all these opportunities happened to you. (
40:03
)
Alexey
: How would you encourage people to still try? It's unlikely if I just do a dataset, it will become trending. I've uploaded quite a few datasets on Kaggle, and none of them trend. Even if I had 10 or 20 upvotes, it didn't make much difference. (
40:03
)
Lavanya
: Yeah, definitely. I mentioned luck because, as you said, at that time-during COVID-people were really active on Kaggle. Maybe the timing was luck, but it wasn't a random decision to just get up one day and scrape Google Play Store. (
41:13
)
I also compared it with the Apple App Store datasets and saw there were datasets for that, but not for Google Play. So there was definitely something missing. It wasn't just a matter of writing some code and it worked like magic. There were bots blocking me, and I got a lot of emails and notifications saying this wasn't allowed.
: I found my way out. I spoke to a lot of legal people. One of my friends told me I needed to license it properly, otherwise I could get in trouble. (
41:58
)
Alexey
: It was a lot of work. (
42:44
)
Lavanya
: Yeah, it was not just scraping and uploading it. There's luck, but there's also analytical thought that went into why I wanted to do this. (
42:44
)
Alexey
: Yeah, I think this is even more important than the result. I mean, yes, it got trending and more people found you, but it's because of the effort you put in. You would probably put a similar amount of effort into a new project, and then one way or another- (
42:56
)
Lavanya
: Yeah, and you can't predict these things. You just can't predict them. You just have to do it, and if it happens, it happens. (
43:19
)
Alexey
: Yeah, I just checked my Kaggle datasets. The largest one has 171 upvotes, which I think is decent. It was a dataset with images of different clothing-pants, shorts, t-shirts-doing some image classification back then. (
43:32
)
Lavanya
: Yeah, I actually might have-I don't know, because at some point I was also building a fashion recommender system just as a pet project. So I did look into a lot of these clothing datasets on Kaggle. I might be one of the upvoters because I downloaded a ton of datasets from Kaggle on clothes. (
43:51
)
Alexey
: The reason I had this dataset-it's not like I just woke up one day and thought, "Let me do a dataset with clothing items." There was a reason. I was writing a book and needed to include pictures of datasets in the book, and all the other clothing datasets were scraped. (
44:17
)
Alexey
: If I used a dataset from Amazon or Zalando or some other company, then the images belonged to them. If I printed them in a book, I might get into trouble. So I thought I needed to create a dataset with a good license that allowed me to do this kind of thing. (
44:17
)
Lavanya
: Yeah, one of my friends was into a lot of open-source contributions at that time, and he pointed out that there are these five types of licenses. He told me to quickly put a license on my dataset, otherwise I could get into trouble. I remember discussing it with him in detail. (
45:06
)
Alexey
: Yeah, so we have a few questions. How can a career pivoter-somebody who's changing careers, a career changer without a computer science degree or main background-break into data? (
45:24
)
Lavanya
: Yeah, I think there are a lot of data roles, at least in the industry that I know. I'm in Seattle, which is a hub for Microsoft and Amazon. A lot of people here are from non-CS backgrounds, including a couple of my friends, all working for these big tech giants in data roles. (
45:42
)
If you're completely non-technical, a technical product manager is something people find easier to get into. You have business acumen, and all you need is to know a little about how software engineering works-not actual coding, but just how SQL works or things like that. That's a very good entry point.
: My friend did civil engineering, did his master's in civil engineering, and then became a senior tech product manager at Amazon. Product manager is one thing. If you're early in your career, BI roles are also quick to get into. (
46:06
)
If you're completely non-technical, a technical product manager is something people find easier to get into. You have business acumen, and all you need is to know a little about how software engineering works-not actual coding, but just how SQL works or things like that. That's a very good entry point.
: You can learn some skills, like Tableau or basic SQL, and just get going. Self-learning is crucial, but those roles still require effort because you need to spend time learning those technologies. But a product manager is easier to get through. (
46:06
)
Alexey
: Yeah, and we talked about Eugene. Eugene also lives in Seattle. I didn't realize you were from the same city. But it's a big city, right? It's not like you can randomly run into people on the street. (
47:27
)
Lavanya
: Yeah, I mean, I'm very close to the area where all these Amazon offices are. Every day, I come across someone familiar, either from my undergrad, master's, or just from LinkedIn. Very active people here. (
47:42
)
Alexey
: So another question: can you please highlight mentorship or networking opportunities which were helpful? (
48:00
)
Lavanya
: Is this about getting those opportunities to be a mentor? (
48:07
)
Alexey
: The way I interpret this question is, you had some mentorship opportunities, but then some of them ended up being helpful for you in some way. (
48:13
)
Lavanya
: Oh, I see. Yeah, I think getting those mentor roles is typically just through reaching out-cold emailing, LinkedIn messages, things like that. (
48:28
)
Lavanya
: Once you build a certain rapport with people in these huge communities, it becomes word of mouth. If you've done good networking and put a good name to your work, it becomes word of mouth, and that's how you excel in these community-driven programs. (
48:28
)
Alexey
: Actually, speaking of our earlier conversation, I scrolled up and saw that we talked about mentorship. The reason I asked you about that is because I was doing some mentorship myself back then, so I was interested in your perspective. (
49:16
)
For me, what was helpful is just structuring my knowledge around things. People come to you with requests and you try to help, and when you say things out loud, it helps structure these things in your head. Then it becomes easier to use this information in other settings. That was the main highlight for me.
: And when people, one year after that, say, "Hey, thanks-because of the interview or the session we did together, now I work at this company. Thanks a lot, you changed my life." That's so cool. (
49:35
)
Lavanya
: Yeah, I think that kind of exposure I only got with my open-ended mentorship experiences. Since you work independently, you had a lot of this, but I was tied to more formal, organized structures with these different groups I was working with. (
50:19
)
There were set expectations and goals, less open-ended stuff. But there's more fun in the open-ended stuff because, as you say, it's just a normal conversation. It's not like I'm going to give you some magic tricks that will change your life.
: But when you hear back from people some years down the line that it helped, that's a really nice feeling that mentorship rewards you with. (
50:31
)
Alexey
: Thank you. So, another question in relation to successful datasets. I wanted to ask, Lavanya, what you consider to be another important element of a portfolio for data science. Basically, the question is: a dataset is one option, but what else, or what instead of a dataset, can we include in our portfolio? (
51:07
)
Building a strong data science portfolio
Lavanya
: Oh yes. I think the dataset itself-I don't think it's even mentioned on my resume. Once you go out for these roles, these are like your pet projects. It's nice to show your curiosity or talk about it at the end of the interview, but it's hardly going to get you the job per se, because those requirements are different. (
51:28
)
To build your portfolio, one thing is you can stand out in the community through these pet projects and extra stuff. But if you're talking about a portfolio in terms of targeting jobs, that's a different kind of effort.
: Maybe this requires a follow-up question or clarification on what portfolio you're talking about: is it just standing out or building networking opportunities, or is it more targeted at job applications? Maybe let's cover both. (
51:53
)
Alexey
: If we want both-if we want to target networking, how should the portfolio look, and if we want to target job opportunities, how should it look? (
52:28
)
Lavanya
: Yeah, okay, sure. Let's talk about just the community-building, networking aspect first. Honestly, none of the things that I did gave me any jobs or roles. I've obviously just worked with two companies, but none of them got me anything from my extra stuff. (
52:42
)
In terms of job building, I think that's very different. In my experience, at least in the US economy, it's highly competitive.
: Profile building for job search is going to be a lot of LeetCode and a lot of conceptual drilling, going deep into the concepts so you can answer all interview questions precisely-no beating around the bush. And then mock interviews. (
54:33
)
Alexey
: But this is more for preparation to pass the interview. But I guess when it comes to portfolio, there are some projects that could be… (
55:14
)
Alexey
: So, I don't know, let's say if I want to work at Amazon, then I can think, "What are the problems that Amazon solves?" and then do a project. Maybe a recommender system, or I know search. (
55:14
)
Maybe Amazon is a bad example because it's so huge and there is everything.
: Maybe there's a smaller e-commerce company, and you can get some e-commerce dataset and do a project on search. Especially if you want to work in this area, when you have an interview with them, you can say, "I maybe don't have professional experience, but I did this project on search, which we can talk about," and then all of a sudden you have things to discuss in the interview. (
55:32
)
Lavanya
: For sure. I would just add to that, sometimes when I also interview people for roles, we do value some industrial experience more than pet projects, only because of scale. (
56:09
)
Lavanya
: When you are doing something on your own, there is no real testing or feedback from actual users. You can just come and tell me, "I got 90% accuracy," but that's not verified. (
56:09
)
I understand if you're starting out new, it's a vicious circle-you want to get into the industry, and then I'm asking you to get industry experience to be able to get into the industry.
: What I'm trying to say is, you can still do pet projects, but be associated with some organization. At my time, I was looking into this organization called OMD. (
56:28
)
They do small projects, but they have industry experts looking over your projects. They organize students and people into data scientists, project managers, industry experts, and all of them work on a single project apart from their regular roles.
: That's a good way to showcase that you are still building your portfolio, but it's not completely your own project. There are others looking into it, equally invested, and it has some actual impact in the real world. (
56:56
)
They do small projects, but they have industry experts looking over your projects. They organize students and people into data scientists, project managers, industry experts, and all of them work on a single project apart from their regular roles.
: There are many organizations like that who do these things. So for applications targeted at job building, I think that's a nice way to build your portfolio. (
56:56
)
Alexey
: Yeah, amazing. And on this note, I realize that I'm late for another meeting, so I need to run. (
57:46
)
Alexey
: But that was an amazing conversation, Lavanya, so thanks a lot for agreeing to this interview. I had a lot of fun talking to you and also some memories from a couple of years ago. Thanks a lot for finding time to do this, and I wish you success. (
57:46
)
Alexey
: Yeah, thank you so much. I'm sure we're going to stay in touch-no more nostalgia chats! And to everyone, thanks for joining us today, for asking questions, for being active. Enjoy the rest of your week. (
58:10
)
Subscribe to our weekly newsletter and
join our Slack
.
We'll keep you informed about our events, articles, courses, and everything else happening in the Club.
Email
Join
DataTalks.Club.
Hosted on
GitHub Pages
.
We use cookies.