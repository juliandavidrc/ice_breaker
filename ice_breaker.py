import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


from third_parties.gist_profile import gist_response_lkprof

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter_data import scrape_user_tweets
from output_parsers import PersonIntel, person_intel_parser


def ice_break(name:str) -> PersonIntel:

    print(os.environ["PROXYCURL_API_KEY"])

    linkedin_profile_url = linkedin_lookup_agent(name="Eden Marco Udemy")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    
    #####  Take Json Linkedin Profile 
    #linkedin_data = gist_response_lkprof()
    #####
    
    #####  Real deploying for Twitter Extraction 

    #twitter_username = twitter_lookup_agent(name=name)
    #tweets = scrape_user_tweets(username=twitter_username, num_tweets=5)
    #####
    
    tweets = scrape_user_tweets(username=name, num_tweets=100)


    
    summary_template = """
         given the Linkedin information {linkedin_information} and twitter {twitter_information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
         3. A topic that may interest them
         4. 2 creative Ice breakers to open a conversation with them
        \n{format_instructions} 
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
        partial_variables={
        "format_instructions": person_intel_parser.get_format_instructions(),
        },
    )


    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)    

    result = chain.run(linkedin_information=linkedin_data, twitter_information=tweets)
    return person_intel_parser.parse(result)



if __name__ == "__main__":
    print("Hello Langchain!")
    name = "Eden Marco"
    result = ice_break(name=name)
