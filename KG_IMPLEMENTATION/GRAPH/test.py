from pipeline import kg_rag_pipeline
import textwrap 

while True:
    try:
        q = input("\n Ask a question: ").strip()
        
        if not q:
            continue
            
        if q.lower() in ['exit', 'quit', 'q']:
            print("\n bye!")
            break
        
        answer = kg_rag_pipeline(q)
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║  ANSWER" + " "*58 + "║")
        print("╠" + "═"*68 + "╣")
        # Word wrap the answer nicely
        wrapped_answer = textwrap.fill(answer, width=66, initial_indent="║  ", subsequent_indent="║  ")
        print(wrapped_answer)
        print("╚" + "═"*68 + "╝")
        
    except KeyboardInterrupt:
        print("\n\n bye!")
        break
    except Exception as e:
        print(f"\n An unhandled error occurred: {e}")