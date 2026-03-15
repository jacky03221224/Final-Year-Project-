<h1>Programs</h1>

<div>
<h2>data_cleaning.py</h2>
<p>This is a program based on the data cleaning prompt.</p>
<p>It performs the following task:
<ol>
<li>Remove rows with missing values</li> 
<li>Remove all the Zacks.com promotion content</li> 
<li>Remove identical news entries</li> 
<li>Formatting</li> 
<ol></p>
</div>

<div>
<h2>relevance.py.py</h2>
<p>This is a program based on the relevance prompt.</p>
<p>It requires a config file and a CSV file as input.</p>
<p>It uses keyword matching to assign labels and scores to each financial news data.</p>
<p>We did not choose to use any Machine learning model because we do not have time for labeling the dataset manually.</p>
<p>It ranges from [0,1], where:
<ol>
<li>Directly Related: 0.70 - 1.00</li> 
<li>Indirectly Related: 0.30 - 0.69</li> 
<li>Unrelated: 0.00 - 0.29</li> 
<ol></p>
</div>
