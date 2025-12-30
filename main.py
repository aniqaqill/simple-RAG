import sys
from src.rag import RAGPipeline
from src.utils import load_text_file

def main():
    # Initialize Pipeline
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return

    # Load Data
    try:
        dataset = load_text_file('dataset/cat-facts.txt')
        print(f'Loaded {len(dataset)} cat facts.')
    except FileNotFoundError:
        print("Error: cat-facts.txt not found. Creating a default one in memory.")
        dataset = ["Cats spend 70% of their lives sleeping.", "A group of cats is called a clowder."]
    
    # Index Data
    try:
        pipeline.index_documents(dataset)
    except Exception as e:
        print(f"Error indexing documents: {e}")
        return

    # Interaction Loop
    print("\n--- RAG Chatbot Ready ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_input = input('\nAsk a question about cats: ').strip()
            if user_input.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            # Retrieve
            retrieved_knowledge = pipeline.retrieve(user_input)
            
            print('\n--- Retrieved Knowledge ---')
            for chunk, similarity in retrieved_knowledge:
                print(f' - (Sim: {similarity:.2f}) {chunk}')

            print('\n--- Chatbot Response ---')
            stream = pipeline.generate_response(user_input, retrieved_knowledge)
            
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
