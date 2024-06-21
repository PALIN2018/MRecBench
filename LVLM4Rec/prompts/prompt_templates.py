templates = {
# LVLM as Recommender, only item pic
     's-1-image': """
This image shows {} items from user's purchase history. The items are arranged in order of purchase, from left to right. 
There are {} candidate items in the item pool: {}.
Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history.
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "explanation", and "recommendations". It follows this structure: 
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate products that are not in the given item pool.].
""",
# LVLM as Recommender, item pic + item title
     's-1-title-image': """
This image shows {} items from user's purchase history. The items are arranged in order of purchase, from left to right. 
The side information of these purchased items are as follows:{}. It's important to note that the order of this side information aligns with the sequence of items in the image. 
There are {} candidate items in the item pool: {}.
Considering the visuals and accompanying side information of the purchased items, Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history.
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "explanation", and "recommendations". It follows this structure: 
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate products that are not in the given item pool.].
""",
# LVLM as Recommender, only item tilles
     's-1-title': """
The user has purchased {} items in chronological order:{}. 
There are {} candidate items in the item pool: {}.
Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history. 
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "explanation", and "recommendations". It follows this structure: 
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.].
""",
# LVLM as item enhancer
     's-2': """
What's in this image?
""",
# LVLM as reranker
     's-3': """
This image shows {} items from user's purchase history. The items are arranged in order of purchase, from left to right. 
The side information of these purchased items are as follows:{}. It's important to note that the order of this side information aligns with the sequence of items in the image. 
This is a pre-ranked item recommendation sequence in order of likelihood that the user will purchase them, from highest to lowest:{}.
Please re-rank these {} candidate items based on the provided purchase history and the pre-ranked item recommendation sequence. 
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "explanation", and "recommendations". It follows this structure: 
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate products that are not in the given item pool.].
""",
# LVLM as item enhancer and recommender
     's-4': """
The user has purchased {} items (including title and description) in chronological order: {}.
There are {} candidate items in the item pool: {}.
Please rank these {} candidate items in order of likelihood that the user will purchase them, from highest to lowest, based on the provided purchase history. 
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "explanation", and "recommendations". It follows this structure:
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.].
""",
# LVLM as item enhancer and reranker
     's-5': """
The user has purchased {} items (including title and description) in chronological order: {}.
This is a pre-ranked item recommendation sequence in order of likelihood that the user will purchase them, from highest to lowest:{}.
Please re-rank these {} candidate items based on the provided purchase history and the pre-ranked item recommendation sequence. 
Do not search on the Internet.
Output format: Please output directly in JSON format, including keys "explanation", and "recommendations". It follows this structure:
- explanation: [Brief explanation of the rationale behind the recommendations]. 
- recommendations: [A list of {} items ranked by likelihood, from highest to lowest. Do not explain each item and just show its title. Do not generate items that are not in the given item pool.].
"""
}
