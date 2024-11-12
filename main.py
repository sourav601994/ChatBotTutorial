from chatbot import get_chatbot_ai_output


if __name__ == "__main__":
    input_query = input("What you want to know: ")
    print(f"AI : ${get_chatbot_ai_output(input_query)[0]}")
    input_query1 = input("What you want to know: ")
    print(f"AI : ${get_chatbot_ai_output(input_query1)[1]}")
    input_query2 = input("What you want to know: ")
    print(f"AI : ${get_chatbot_ai_output(input_query2)[2]}")
    

#"Employees are eligible for how many paid leaves in a year?"