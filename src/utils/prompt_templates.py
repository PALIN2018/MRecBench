templates = {
    # only item pic (concatenated), no title
     'r-1': """
This image shows {} items from user's purchase history. The items are arranged in order of purchase, from left to right. 
There are {} candidate items in the item pool: {}.
Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history.
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "purchased_items", "explanation", and "recommendations". It follows this structure: 
- purchased_items: [List of items the user has purchased].
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate products that are not in the given item pool.].
""",
# item pic + item title
     'r-2': """
This image shows {} items from user's purchase history. The items are arranged in order of purchase, from left to right. 
The side information of these purchased items are as follows:{}. It's important to note that the order of this side information aligns with the sequence of items in the image. 
There are {} candidate items in the item pool: {}.
Considering the visuals and accompanying side information of the purchased items, Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history.
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "purchased_items", "explanation", and "recommendations". It follows this structure: 
- purchased_items: [List of items the user has purchased].
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate products that are not in the given item pool.].
""",
# only item tilles, no item pic
     'r-3': """
The user has purchased {} items in chronological order:{}. 
There are {} candidate items in the item pool: {}.
Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history. 
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "purchased_items", "explanation", and "recommendations". It follows this structure: 
- purchased_items: [List of items the user has purchased].
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.].
""",
     't-1': """
The user has purchased {} items in chronological order:{}. 
There are {} candidate items in the item pool: {}.
Please use the Censydiam user motivation analysis model to analyze this user's purchase history, and rank these {} candidates from high to low based on the likelihood of the user's purchase according to the analysis results.
Do not search on the Internet. Do not use code interpreter.
Output format: Please output directly in JSON format, including keys "purchased_items", "explanation", and "recommendations". It follows this structure: 
- purchased_items: [List of items the user has purchased].
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.].
""",
     't-2': """
The user has purchased {} items in chronological order:{}. 
There are {} candidate items in the item pool: {}.
Please use the Sheth-Newman-Gross Consumer Value Model to analyze this user's purchase history, and rank these {} candidates from high to low based on the likelihood of the user's purchase according to the analysis results.
Do not search on the Internet. Do not use code interpreter.
Output format: Please output directly in JSON format, including keys "purchased_items", "explanation", and "recommendations". It follows this structure: 
- purchased_items: [List of items the user has purchased].
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.].
""",
     't-3': """
Please use the Censydiam model for user motivation analysis, based on the following purchase history:{}, to analyze the user's behavior. Output requirements: a simple paragraph to describe the user motivation. You need to mention items and do not include other words.
""",
# TODO: support multiple images
}
