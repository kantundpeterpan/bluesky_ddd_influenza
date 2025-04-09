Presentation Script: Beyond Twitter - Exploring bluesky.social for Digital Disease Detection

[SLIDE 1: Title Slide]
Good morning/afternoon, everyone. My name is Heiner Atze, and this presentation is for the Digital Epidemiology course at Hasselt University. Today, I'll be discussing a project exploring the potential of bluesky.social as a data source for digital disease detection, specifically focusing on influenza-like illness (ILI) surveillance.

[Transition: Briefly gesture towards the agenda.]

[SLIDE 2: Outline]
Here’s the outline: I will give an introduction to the bluesky social network, then discuss accessing its data via the bluesky API, and finally cover the extraction and analysis of ILI-related bluesky messages.

[Transition: Set the stage for the project.]

[SLIDE 3: Introduction]
So, let's start with the basics - the bluesky social network.

[SLIDE 4: bluesky: General Aspects]
bluesky is similar to twitter, but differs in more than just user experience. It is a microblogging platform but with a decentralized and open-source approach. This has implications for data accessibility and algorithmic transparency, which are crucial for reliable digital epidemiology.

[SLIDE 5: Decentralization and Democratization of content algorithms]
bluesky uses Decentralized Identifiers and Personal Data Servers. DIDs are immutable, so that user handles are stable.  More crucially, users are able to choose, prioritize and develop feed generators and content labelers. This means a greater chance to democratize the data and content that is important to them.

[SLIDE 6: Development of user activity]
bluesky has already reached a significant user base, around 33 million users. User base expansions appear to coincide with pivotal events. This suggests that bluesky is increasingly becoming an important venue for rapid public opinion formation. It is important that epidemiology and digital disease detection research keeps an eye on the development of this new venue.

[SLIDE 7: Literature addressing bluesky]
A quick search finds some publications that have addressed bluesky. They cover mostly migration from X and architectural consideration, but, importantly, none mention epidemiology or digital disease detection. With this project I want to illustrate that this platform has potential.

[Transition: Now focusing on data access.]

[SLIDE 8: Exploration of bluesky data]
Let's delve into how we can actually access and utilize the data from bluesky.

[SLIDE 9: bluesky API]
The bluesky API is publicly accessible and free, making it attractive for research. The API is well-documented.

[SLIDE 10: searchPosts API method]
The workhorse of the current project is the `searchPosts` API method. `searchPosts` allows deterministic search, which is important for exhaustive sampling.
There are some quirks concerning programmatic access as it return 100 posts at most.

[SLIDE 11: getProfiles]
For completeness, `getProfiles` are also available, but they are not used in this project.

[SLIDE 12: Post metadata]
Post metadata is extensive. The PostView object that is returned is rich in information, such as post URI, author information, the text of the message, and attached media. Note that geoinformation is missing from the metadata.

[SLIDE 13: User information]
User information is also limited in that geographic information is completely missing, but feed generators and labelers are available.

[SLIDE 14: Project]
So what is the project? With this project I want to explore `bluesky` post data for digital disease surveillance, and create a continuous surveillance pipeline.

[SLIDE 15: Outline]
This project attempts to show the potential to show that a continuous surveillance pipeline can be constructed.

[SLIDE 16: Data extraction]
Let's talk about how I extracted the data and what data I used.

[SLIDE 17: ILI symptom related message extraction]
The data extraction focuses on french `bluesky` posts due to data volume constraint. A list of keywords are used to extract ILI symptom related messages. Complete message data and time series counts are extracted.
[NOTE FOR YOURSELF: You can mention some of the limitations and assumptions here, like time zone issues and negligible data from French-speaking African countries.]

[SLIDE 18: Basal network activity]
Basal network activity data informs on the general activity of the network.

[SLIDE 19: Case data]
Finally, publicly available historical data is retrieved for later modeeling.

[Transition: Leading into the results section – time to show the findings!]

[SLIDE 20: Results]

[SLIDE 21: Post count time series]
Let's see the results on post count time series.

[SLIDE 22: Raw posts counts]
Here are the raw post counts, analysed from August 1st, 2023.

[SLIDE 23: Keyword posts *vs.* ILI incidence]
I compared Grippe keyword post with the case incidence. The correllation results indicated that ILI posts and ILI incidence were highly correlated. In fact, the correlation is higher for ILI posts, which already indicates signal improvement just by querying for ILI post keywords.

[SLIDE 24: Normalized keyword posts *vs.* ILI incidence]
I normalized the ILI keyword posts using the number of control messages which should control for the changes in users active on the network. The seasonality is still apparent,
the correlation decreased.
You can see the peaks are of the same magnitude, something that I found to be unexpected given that the 2023 / 2024 season was moderate compared to the current ILI epidemic and one would expect to find proportionally more messages about ILI recently.

[Transitions into the Machine Learning approach]
I'll now talk on how with the data, machine learning and machine language approaches are implemented.

[SLIDE 25: Machine learning]

[SLIDE 26: Features]
Several features are used: (1) amount of control posts, (2) number of posts containing ILI related keywords, (3) time and seasonal features and (4) lag terms. All features are aggregated by the week.

[SLIDE 27: Gradient boosted trees]
Gradient boosted trees are powerful tools to model time series. It is a sequential method, corrects errors, uses weighted averaging and robust to outliers. This is the approach undertaken here.

[SLIDE 28: Model evaluation]
To validate the model, a Time series split validation is undertaken. The purpose for this method is that it retains temporal information and mimics data acquisition better.

[SLIDE 29: Predictions and metrics]
These are the predictions and metrics. As can be seen from MAE values, a huge generalization error can be found.

[SLIDE 30: Permutation importance]
Permutation importance helps understand the most important features of the model.

[Transition: Leading into LLM approach]
Can AI help improve the accuracy of the mode? Lets see....

[SLIDE 31: Can AI help?]

[SLIDE 32: Idea]
A large language model is used to filter the posts.

[SLIDE 33: Prompt and output]
The propmt to the model is provided and the JSON model output will be returned. It includes the indication of the post to ILI related and an arria of symptoms.

[SLIDE 34: Examples]
Here are some examples of extraction for ili-positive and ili-negative cases.

[SLIDE 35: LLM annotated post counts]
Finally, here is the post counts with LLM annotations.

[SLIDE 36: Conclusion]
In conclusion: I believe that `bluesky` = promising data source and more data is needed.

[SLIDE 37: Conclusion]
Finally, a discussion on future actions to take.

[SLIDE 38: Pipeline (WIP)]
Here finally is a preview of the data engineering continuous data acquisition pipeline that the project undertakes. I would like to thank all the open source tools for this pipeline and I will publish the code repository after this presentation.

[SLIDE 39: Bibliography]

[Final Statements]
Thank you for your time and attention. I'm now happy to answer any questions you may have.