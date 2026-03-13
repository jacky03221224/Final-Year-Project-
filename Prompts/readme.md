<h1>Prompts used in calculating the sentiment score</h1>
<h2>Reminders</h2>
<p>** When using the prompt, you should replace &lt;Company Name&gt; and  &lt;Company Ticker&gt; with the actual company's name and ticker.</p>

<div>
<h2>Relevance Prompt</h2>
<p>The relevance_prompt.docx file is used to check the relevance of the news to the Company.</p>
</div>

<div>
<h2>Directional Sentiment Prompt</h2>
<p>The directional_sentiment_prompt.docx file is used to determine the directional sentiment of the news.</p>
</div>

<h1>Programs</h1>
<h2>data_cleaning.py</h2>
<p>This is a program based on the data cleaning prompt.</p>
<p>It performs the following task:
<ol>
<li>Remove rows with missing values</li> 
<li>Remove all the Zacks.com promotion content</li> 
<li>Remove identical news entries</li> 
<li>Formatting</li> 
<ol></p>
