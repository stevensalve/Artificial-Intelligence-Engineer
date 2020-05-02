<h1>Artificial Intelligence Engineer Assignments </h1>

<h3>Data Science with Python</h3> 
    <b>Assignments - All Dataset are in this folder</b><br>
        <ul>
               <li>Customer Service Requests Analysis</li>
               <li>Movielens Case Study</li>
               <li>Building user-based recommendation model for Amazon.</li>
               <li>Comcast Telecom Consumer Complaints.</li>
               <li>Retail Analysis with Walmart Data</li>
        </ul>

<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" id="34">
<h3>Project 1  : Customer Service Requests Analysis</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p><strong>Background of Problem Statement :</strong></p>

<p>NYC 311's mission is to provide the public with quick and easy access to all New York City government services and information while offering the best customer service. Each day, NYC311 receives thousands of requests related to several hundred types of non-emergency services, including noise complaints, plumbing issues, and illegally parked cars. These requests are received by NYC311 and forwarded to the relevant agencies such as the police, buildings, or transportation. The agency responds to the request, addresses it, and then closes it.</p>

<p><strong>Problem Objective :</strong></p>

<p>Perform a service request data analysis of New York City 311 calls. You will focus on the data wrangling techniques to understand the pattern in the data and also visualize the major complaint types.<br>
Domain: Customer Service</p>

<p><strong>Analysis Tasks to be performed:</strong></p>

<p>(Perform a service request data analysis of New York City 311 calls)&nbsp;</p>
<ol>
	<li>Import a 311 NYC service request.</li>
	<li>Read or convert the columns ‘Created Date’ and Closed Date’ to datetime datatype and create a new column ‘Request_Closing_Time’ as the time elapsed between request creation and request closing. (Hint: Explore the package/module datetime)</li>
	<li>Provide major insights/patterns that you can offer in a visual format (graphs or tables); at least 4 major conclusions that you can come up with after generic data mining.</li>
	<li>Order the complaint types based on the average ‘Request_Closing_Time’, grouping them for different locations.</li>
	<li>Perform a statistical test for the following:</li>
</ol>

<p>Please note: For the below statements you need to state the Null and Alternate and then provide a statistical test to accept or reject the Null Hypothesis along with the corresponding ‘p-value’.</p>
<ul>
	<li>Whether the average response time across complaint types is similar or not (overall)</li>
	<li>Are the type of complaint or service requested and location related?</li>
</ul>
<p>Dataset Description :</p>
<div>
<table class="table" style="width:100%">
	<tbody>
		<tr>
			<td><strong>Field</strong></td>
			<td><strong>Description</strong></td>
		</tr>
		<tr>
			<td>Unique Key</td>
			<td>(Plain text) - Unique identifier for the complaints</td>
		</tr>
		<tr>
			<td>Created Date</td>
			<td>(Date and Time) - The date and time on which the complaint is raised</td>
		</tr>
		<tr>
			<td>Closed Date</td>
			<td>(Date and Time) &nbsp;- The date and time on which the complaint is closed</td>
		</tr>
		<tr>
			<td>Agency</td>
			<td>(Plain text) - Agency code</td>
		</tr>
		<tr>
			<td>Agency Name</td>
			<td>(Plain text) - Name of the agency</td>
		</tr>
		<tr>
			<td>Complaint Type</td>
			<td>(Plain text) - Type of the complaint</td>
		</tr>
		<tr>
			<td>Descriptor</td>
			<td>(Plain text) - Complaint type label (Heating - Heat, Traffic Signal Condition - Controller)</td>
		</tr>
		<tr>
			<td>Location Type</td>
			<td>(Plain text) - Type of the location (Residential, Restaurant, Bakery, etc)</td>
		</tr>
		<tr>
			<td>Incident Zip</td>
			<td>(Plain text) - Zip code for the location</td>
		</tr>
		<tr>
			<td>Incident Address</td>
			<td>(Plain text) - Address of the location</td>
		</tr>
		<tr>
			<td>Street Name</td>
			<td>(Plain text) - Name of the street</td>
		</tr>
		<tr>
			<td>Cross Street 1</td>
			<td>(Plain text) - Detail of cross street</td>
		</tr>
		<tr>
			<td>Cross Street 2</td>
			<td>(Plain text) - Detail of another cross street</td>
		</tr>
		<tr>
			<td>Intersection Street 1</td>
			<td>(Plain text) - Detail of intersection street if any</td>
		</tr>
		<tr>
			<td>Intersection Street 2</td>
			<td>(Plain text) - Detail of another intersection street if any</td>
		</tr>
		<tr>
			<td>Address Type</td>
			<td>(Plain text) - Categorical (Address or Intersection)</td>
		</tr>
		<tr>
			<td>City</td>
			<td>(Plain text) - City for the location</td>
		</tr>
		<tr>
			<td>Landmark</td>
			<td>(Plain text) - Empty field</td>
		</tr>
		<tr>
			<td>Facility Type</td>
			<td>(Plain text) - N/A</td>
		</tr>
		<tr>
			<td>Status</td>
			<td>(Plain text) - Categorical (Closed or Pending)</td>
		</tr>
		<tr>
			<td>Due Date</td>
			<td>(Date and Time) - Date and time for the pending complaints</td>
		</tr>
		<tr>
			<td>Resolution Action Updated Date</td>
			<td>(Date and Time) - Date and time when the resolution was provided</td>
		</tr>
		<tr>
			<td>Community Board</td>
			<td>(Plain text) - Categorical field (specifies the community board with its code)</td>
		</tr>
		<tr>
			<td>Borough</td>
			<td>(Plain text) - Categorical field (specifies the community board)</td>
		</tr>
		<tr>
			<td>X Coordinate</td>
			<td>(State Plane) (Number)</td>
		</tr>
		<tr>
			<td>Y Coordinate</td>
			<td>(State Plane) (Number)</td>
		</tr>
		<tr>
			<td>Park Facility Name</td>
			<td>(Plain text) - Unspecified</td>
		</tr>
		<tr>
			<td>Park Borough</td>
			<td>(Plain text) - Categorical (Unspecified, Queens, Brooklyn etc)</td>
		</tr>
		<tr>
			<td>School Name</td>
			<td>(Plain text) - Unspecified</td>
		</tr>
		<tr>
			<td>School Number</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School Region</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School Code</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School Phone Number</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School Address</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School City</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School State</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School Zip</td>
			<td>(Plain text) &nbsp;- Unspecified</td>
		</tr>
		<tr>
			<td>School Not Found</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>School or Citywide Complaint</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Vehicle Type</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Taxi Company Borough</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Taxi Pick Up Location</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Bridge Highway Name</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Bridge Highway Direction</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Road Ramp</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Bridge Highway Segment</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Garage Lot Name</td>
			<td>(Plain text) &nbsp;- Empty Field<br>
			&nbsp;</td>
		</tr>
		<tr>
			<td>Ferry Direction</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Ferry Terminal Name</td>
			<td>(Plain text) &nbsp;- Empty Field</td>
		</tr>
		<tr>
			<td>Latitude</td>
			<td>(Number) - Latitude of the location</td>
		</tr>
		<tr>
			<td>Longitude</td>
			<td>(Number) - Longitude of the location</td>
		</tr>
		<tr>
			<td>Location</td>
			<td>(Location) - Coordinates (Latitude, Longitude)</td>
		</tr>
	</tbody>
</table>
</div>

<br>
<hr>
<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" >
<h3>Project 2  : Movielens Case Study</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p><strong>Background of Problem Statement :</strong></p>


<p>The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. Members of the GroupLens Research Project are involved in many research projects related to the fields of information filtering, collaborative filtering, and recommender systems. The project is led by professors John Riedl and Joseph Konstan. The project began to explore automated collaborative filtering in 1992 but is most well known for its worldwide trial of an automated collaborative filtering system for Usenet news in 1996. Since then the project has expanded its scope to research overall information by filtering solutions, integrating into content-based methods, as well as, improving current collaborative filtering technology.</p>

<p><strong>Problem Objective :</strong></p>

<p>Here, we ask you to perform the analysis using the Exploratory Data Analysis technique. You need to find features affecting the ratings of any particular movie and build a model to predict the movie ratings.</p>

<p><strong>Domain</strong>: Entertainment</p>

<p><strong>Analysis Tasks to be performed:</strong></p>

<ul>
	<li>Import the three datasets</li>
	<li>Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating. (Hint: (i) Merge two tables at a time. (ii) Merge the tables using two primary keys MovieID &amp; UserId)</li>
	<li>Explore the datasets using visual representations (graphs or tables), also include your comments on the following:</li>
</ul>

<ol style="margin-left:40px">
	<li>User Age Distribution</li>
	<li>User rating of the movie “Toy Story”</li>
	<li>Top 25 movies by viewership rating</li>
	<li>Find the ratings for all the movies reviewed by for a particular user of user id = 2696</li>
</ol>

<ul>
	<li>Feature Engineering:</li>
</ul>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Use column genres:</p>

<ol style="margin-left:40px">
	<li>Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres)</li>
	<li>Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre.&nbsp;</li>
	<li>Determine the features affecting the ratings of any particular movie.</li>
	<li>Develop an appropriate model to predict the movie ratings</li>
</ol>

<p><strong>Dataset Description :</strong></p>

<p>These files contain 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.</p>

<p><strong>Ratings.dat</strong><br>
&nbsp;&nbsp; &nbsp;Format - UserID::MovieID::Rating::Timestamp</p>

<div>
<table class="table" style="width:100%">
	<tbody>
		<tr>
			<td><strong>Field</strong></td>
			<td><strong>Description</strong></td>
		</tr>
		<tr>
			<td>UserID</td>
			<td>Unique identification for each user</td>
		</tr>
		<tr>
			<td>MovieID</td>
			<td>Unique identification for each movie</td>
		</tr>
		<tr>
			<td>Rating</td>
			<td>User rating for each movie</td>
		</tr>
		<tr>
			<td>Timestamp</td>
			<td>Timestamp generated while adding user review</td>
		</tr>
	</tbody>
</table>
</div>

<ul>
	<li>UserIDs range between 1 and 6040&nbsp;</li>
	<li>The MovieIDs range between 1 and 3952</li>
	<li>Ratings are made on a 5-star scale (whole-star ratings only)</li>
	<li>A timestamp is represented in seconds since the epoch is returned by time(2)</li>
	<li>Each user has at least 20 ratings
	<p>&nbsp;</p>
	</li>
</ul>

<p><strong>Users.dat</strong><br>
Format - &nbsp;UserID::Gender::Age::Occupation::Zip-code</p>

<div>
<table class="table" style="width:100%">
	<tbody>
		<tr>
			<td>Field</td>
			<td>Description</td>
		</tr>
		<tr>
			<td>UserID</td>
			<td>Unique identification for each user</td>
		</tr>
		<tr>
			<td>Genere</td>
			<td>Category of each movie</td>
		</tr>
		<tr>
			<td>Age</td>
			<td>User’s age</td>
		</tr>
		<tr>
			<td>Occupation</td>
			<td>User’s Occupation</td>
		</tr>
		<tr>
			<td>Zip-code</td>
			<td>Zip Code for the user’s location</td>
		</tr>
	</tbody>
</table>
</div>

<p>All demographic information is provided voluntarily by the users and is not checked for accuracy. Only users who have provided demographic information are included in this data set.</p>

<ul>
	<li>Gender is denoted by an "M" for male and "F" for female</li>
	<li>Age is chosen from the following ranges:</li>
</ul>

<p>&nbsp;</p>

<div>
<table class="table" style="width:100%">
	<tbody>
		<tr>
			<td><strong>Value</strong></td>
			<td><strong>Description</strong></td>
		</tr>
		<tr>
			<td>1</td>
			<td>"Under 18"</td>
		</tr>
		<tr>
			<td>18</td>
			<td>"18-24"</td>
		</tr>
		<tr>
			<td>25</td>
			<td>"25-34"</td>
		</tr>
		<tr>
			<td>35</td>
			<td>"35-44"</td>
		</tr>
		<tr>
			<td>45</td>
			<td>"45-49"</td>
		</tr>
		<tr>
			<td>50</td>
			<td>"50-55"</td>
		</tr>
		<tr>
			<td>56</td>
			<td>"56+"</td>
		</tr>
	</tbody>
</table>
</div>

<p>&nbsp;</p>

<ul>
	<li>Occupation is chosen from the following choices:</li>
</ul>

<div>
<table class="table" style="width:100%">
	<tbody>
		<tr>
			<td><strong>Value</strong><br>
			&nbsp;</td>
			<td><strong>Description</strong></td>
		</tr>
		<tr>
			<td>0</td>
			<td>"other" or not specified</td>
		</tr>
		<tr>
			<td>1</td>
			<td>"academic/educator"</td>
		</tr>
		<tr>
			<td>2</td>
			<td>"artist”</td>
		</tr>
		<tr>
			<td>3</td>
			<td>"clerical/admin"</td>
		</tr>
		<tr>
			<td>4</td>
			<td>"college/grad student"</td>
		</tr>
		<tr>
			<td>5</td>
			<td>"customer service"</td>
		</tr>
		<tr>
			<td>6</td>
			<td>"doctor/health care"</td>
		</tr>
		<tr>
			<td>7</td>
			<td>"executive/managerial"</td>
		</tr>
		<tr>
			<td>8</td>
			<td>"farmer"</td>
		</tr>
		<tr>
			<td>9</td>
			<td>"homemaker"</td>
		</tr>
		<tr>
			<td>10</td>
			<td>"K-12 student"</td>
		</tr>
		<tr>
			<td>11</td>
			<td>"lawyer"</td>
		</tr>
		<tr>
			<td>12</td>
			<td>"programmer"</td>
		</tr>
		<tr>
			<td>13</td>
			<td>"retired"</td>
		</tr>
		<tr>
			<td>14</td>
			<td>&nbsp;"sales/marketing"</td>
		</tr>
		<tr>
			<td>15</td>
			<td>"scientist"</td>
		</tr>
		<tr>
			<td>16</td>
			<td>&nbsp;"self-employed"</td>
		</tr>
		<tr>
			<td>17</td>
			<td>"technician/engineer"</td>
		</tr>
		<tr>
			<td>18</td>
			<td>"tradesman/craftsman"</td>
		</tr>
		<tr>
			<td>19</td>
			<td>"unemployed"</td>
		</tr>
		<tr>
			<td>20</td>
			<td>"writer”</td>
		</tr>
	</tbody>
</table>
</div>

<p><br>
<strong>Movies.dat</strong><br>
Format - MovieID::Title::Genres</p>

<div>
<table class="table" style="width:100%">
	<tbody>
		<tr>
			<td>Field</td>
			<td>Description</td>
		</tr>
		<tr>
			<td>MovieID</td>
			<td>Unique identification for each movie</td>
		</tr>
		<tr>
			<td>Title</td>
			<td>A title&nbsp;for each movie</td>
		</tr>
		<tr>
			<td>Genres</td>
			<td>Category of each movie</td>
		</tr>
	</tbody>
</table>
</div>

<p>&nbsp;</p>

<ul>
	<li>&nbsp;Titles are identical to titles provided by the IMDB (including year of release)</li>
</ul>

<p>&nbsp;</p>

<ul>
	<li>Genres are pipe-separated and are selected from the following genres:</li>
</ul>

<ol style="margin-left:40px">
	<li>Action</li>
	<li>Adventure</li>
	<li>Animation</li>
	<li>Children's</li>
	<li>Comedy</li>
	<li>Crime</li>
	<li>Documentary</li>
	<li>Drama</li>
	<li>Fantasy</li>
	<li>Film-Noir</li>
	<li>Horror</li>
	<li>Musical</li>
	<li>Mystery</li>
	<li>Romance</li>
	<li>Sci-Fi</li>
	<li>Thriller</li>
	<li>War</li>
	<li>Western</li>
</ol>
<ul>
	<li>Some MovieIDs do not correspond to a movie due to accidental duplicate entries and/or test entries</li>
	<li>Movies are mostly entered by hand, so errors and inconsistencies may exist</li>
</ul>
<p>Please download the dataset from <a href="https://github.com/Simplilearn-Edu/Data-Science-with-Python-Project-One" target="_blank">here</a>&nbsp;</p>


<br>
<hr>
<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" >
<h3>Project 3  : Building user-based recommendation model for Amazon.</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p>The dataset provided contains movie reviews given by Amazon customers. Reviews were given between May 1996 and July 2014.</p>

<p><strong><u>Data Dictionary</u></strong><br>
UserID – 4848 customers who provided a rating for each movie<br>
Movie 1 to Movie 206 – 206 movies for which ratings are provided by 4848 distinct users</p>

<p><u><strong>Data Considerations</strong></u><br>
- All the users have not watched all the movies and therefore, all movies are not rated. These missing values are represented by NA.<br>
- Ratings are on a scale of -1 to 10 where -1 is the least rating and 10 is the best.</p>

<p><u><strong>Analysis Task</strong></u><br>
- Exploratory Data Analysis:</p>

<ul>
	<li>Which movies have maximum views/ratings?</li>
	<li>What is the average rating for each movie? Define the top 5 movies with the maximum ratings.</li>
	<li>Define the top 5 movies with the least audience.</li>
</ul>

<p>- Recommendation Model: Some of the movies hadn’t been watched and therefore, are not rated by the users. Netflix would like to take this as an opportunity and build a machine learning recommendation algorithm which provides the ratings for each of the users.</p>

<ul>
	<li>Divide the data into training and test data</li>
	<li>Build a recommendation model on training data</li>
	<li>Make predictions on the test data</li>
</ul>
</div></div>

<br>
<hr>
<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" >
<h3>Project 4  : Comcast Telecom Consumer Complaints .</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p>Comcast is an American global telecommunication company. The firm has been providing terrible customer service. They continue to fall short despite repeated promises to improve. Only last month (October 2016) the authority fined them a $2.3 million, after receiving&nbsp;over 1000 consumer complaints.<br>
The existing database will serve as a repository of public customer complaints filed against Comcast.<br>
It will help to pin down what is wrong with Comcast's customer service.</p>

<p><strong><u>Data Dictionary</u></strong></p>

<ul>
	<li>Ticket #: Ticket number assigned to each complaint</li>
	<li>Customer Complaint: Description of complaint</li>
	<li>Date: Date of complaint</li>
	<li>Time: Time of complaint</li>
	<li>Received Via: Mode of communication of the complaint</li>
	<li>City: Customer city</li>
	<li>State: Customer state</li>
	<li>Zipcode: Customer zip</li>
	<li>Status: Status of complaint</li>
	<li>Filing on behalf of someone</li>
</ul>

<p><u><strong>Analysis Task</strong></u></p>

<p>To perform these tasks, you can use any of the different Python libraries such as NumPy, SciPy, Pandas, scikit-learn, matplotlib, and BeautifulSoup.</p>

<p style="margin-left:40px">- Import data into Python environment.<br>
- Provide the trend chart for the number of complaints at monthly and daily granularity levels.<br>
- Provide a table with the frequency of complaint types.</p>

<ul>
	<li>Which complaint types are maximum i.e., around internet, network issues, or across any other domains.</li>
</ul>

<p style="margin-left:40px">- Create a new categorical variable with value as <strong>Open </strong>and <strong>Closed</strong>. Open &amp;&nbsp;Pending is to be categorized as Open and Closed &amp; Solved is to be categorized as Closed.<br>
- Provide state wise status of complaints in a stacked bar chart. Use the categorized variable from Q3. Provide insights on:</p>

<ul>
	<li>Which state has the maximum complaints</li>
	<li>Which state has the highest percentage of unresolved complaints</li>
</ul>

<p style="margin-left:40px">- Provide the percentage of complaints resolved till date, which were received through the Internet and customer care calls.</p>

<p>The analysis results to be provided with insights wherever applicable.</p>


<br>
<hr>
<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" >
<h3>Project  5: Retail A5nalysis with Walmart Data</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p>One of the leading retail stores in the US, Walmart, would like to predict the sales and demand accurately. There are certain events and holidays which impact sales on each day. There are sales data available for 45 stores of Walmart. The business is facing a challenge due to unforeseen demands and runs out of stock some times, due to the inappropriate machine learning algorithm. An&nbsp;ideal ML algorithm will predict demand accurately and ingest factors like economic conditions including CPI, Unemployment Index, etc.</p>

<p>Walmart runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of all, which are the&nbsp;Super Bowl, Labour Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. Part of the challenge presented by this competition is modeling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data. Historical sales data for 45 Walmart stores located in different regions are available.</p>

<p><strong>Dataset Description</strong></p>

<p>This is the historical data that covers sales from 2010-02-05 to 2012-11-01, in the file Walmart_Store_sales. Within this file you will find the following fields:</p>

<ul>
	<li>
	<p>Store - the store number</p>
	</li>
	<li>
	<p>Date - the week of sales</p>
	</li>
	<li>
	<p>Weekly_Sales - &nbsp;sales for the given store</p>
	</li>
	<li>
	<p>Holiday_Flag - whether the week is a special holiday week 1 – Holiday week 0 – Non-holiday week</p>
	</li>
	<li>
	<p>Temperature - Temperature on the day of sale</p>
	</li>
	<li>
	<p>Fuel_Price - Cost of fuel in the region</p>
	</li>
	<li>
	<p>CPI – Prevailing consumer price index</p>
	</li>
	<li>
	<p>Unemployment - Prevailing unemployment rate</p>
	</li>
</ul>

<p><strong>Holiday Events</strong></p>

<p>Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13<br>
Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13<br>
Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13<br>
Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13</p>

<p><strong>Analysis Tasks</strong></p>

<p><strong>Basic Statistics tasks</strong></p>

<ul>
	<li>
	<p>Which store has maximum sales</p>
	</li>
	<li>
	<p>Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation</p>
	</li>
	<li>
	<p>Which store/s has good quarterly growth rate in Q3’2012</p>
	</li>
	<li>
	<p>Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together</p>
	</li>
	<li>
	<p>Provide a monthly and semester view of sales in units and give insights</p>
	</li>
</ul>

<p><strong>Statistical Model</strong></p>

<p>For Store 1 – Build&nbsp; prediction models to forecast demand</p>

<ul>
	<li>
	<p>Linear Regression – Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order). Hypothesize if CPI, unemployment, and fuel price have any impact on sales.</p>
	</li>
	<li>
	<p>Change dates into days by creating new variable.</p>
	</li>
</ul>

<p>Select the model which gives best accuracy.</p>