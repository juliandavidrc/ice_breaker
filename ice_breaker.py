import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


from third_parties.gist_profile import gist_response_lkprof

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter_data import scrape_user_tweets

# information = """Elon Reeve Musk (Pretoria, 28 de junio de 1971), conocido como Elon Musk, es un empresario, inversor y magnate sudafricano que también posee las nacionalidades canadiense y estadounidense. Es el fundador, consejero delegado e ingeniero jefe de SpaceX; inversor ángel, director general y arquitecto de productos de Tesla, Inc.; fundador de The Boring Company; cofundador de Neuralink y OpenAI, aunque ya no tiene más participación en esta última por desacuerdos en el rumbo de la empresa, además de ser el director de tecnología de X Corp..4​ Con un patrimonio neto estimado en unos 207 mil millones de dólares en junio de 2023,5​ Musk es la persona más rica del mundo según el índice de multimillonarios de Bloomberg y la lista de multimillonarios en tiempo real de Forbes.6​7​
# Musk nació de madre canadiense y padre sudafricano blanco, y se crio en Pretoria (Sudáfrica). Estudió brevemente en la Universidad de Pretoria antes de trasladarse a Canadá a los 17 años. Se matriculó en la Universidad de Queen y se trasladó a la Universidad de Pensilvania dos años después, donde se graduó en Economía y Física. En 1995 se trasladó a California para asistir a la Universidad Stanford, pero en su lugar decidió seguir una carrera empresarial, cofundando la empresa de software web Zip2 con su hermano Kimbal. La empresa fue adquirida por Compaq por 307 millones de dólares en 1999. Ese mismo año, Musk cofundó el banco online X.com, que se fusionó con Confinity en 2000 para formar PayPal. La empresa fue comprada por eBay en 2002 por 1500 millones de dólares.
# En 2002, Musk fundó SpaceX, fabricante aeroespacial y empresa de servicios de transporte espacial, de la que es CEO e ingeniero jefe. En 2003, se unió al fabricante de vehículos eléctricos Tesla Motors, Inc. (ahora Tesla, Inc.) como presidente y arquitecto de productos, convirtiéndose en su consejero delegado en 2008. En 2006, ayudó a crear SolarCity, una empresa de servicios de energía solar que posteriormente fue adquirida por Tesla y se convirtió en Tesla Energy. En 2015, cofundó OpenAI, una empresa de investigación sin ánimo de lucro que promueve la inteligencia artificial amigable. En 2016, cofundó Neuralink, una empresa de neurotecnología centrada en el desarrollo de interfaces cerebro-ordenador, y fundó The Boring Company, una empresa de construcción de túneles. También acordó la compra de la importante red social estadounidense Twitter en 2022 por 44 000 millones de dólares. Musk también ha propuesto el hyperloop. En noviembre de 2021, el director general de Tesla fue la primera persona de la historia en acumular una fortuna de 300 000 millones de dólares.8​"""

name = "Eden Marco"
if __name__ == "__main__":
    print("Hello LangChain!")
    # print(os.environ['OPENAI_API_KEY'])
    print(os.environ["PROXYCURL_API_KEY"])

    linkedin_profile_url = linkedin_lookup_agent(name="Eden Marco Udemy")
    linkedin_data = scrape_linkedin_profile(
        # linkedin_profile_url="https://www.linkedin.com/in/harrison-chase-961287118/"
        linkedin_profile_url=linkedin_profile_url
    )

    #####  Real deploying for Twitter Extraction 
    
    #twitter_username = twitter_lookup_agent(name=name)
    #tweets = scrape_user_tweets(username=twitter_username, num_tweets=5)
    #####
    
    #twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=name, num_tweets=100)


    
    summary_template = """
         given the Linkedin information {linkedin_information} and twitter {twitter_information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
         3. A topic that may interest them
         4. 2 creative Ice breakers to open a conversation with them 
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
    )


    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # ** Take linkedin profile from githubuserConnect JSON **
    # linkedin_data = gist_response_lkprof()
    # ** ************************************************* **

    # print(linkedin_data.json())

    print(chain.run(linkedin_information=linkedin_data, twitter_information=tweets))

    #print(scrape_user_tweets(username="@EdenEmarco177"))
