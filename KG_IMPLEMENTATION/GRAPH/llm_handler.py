
from nim_client import NvidiaLLMClient

LLM_API_KEY = "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" 
# LLM_MODEL = "meta/llama-3.2-1b-instruct"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
llm = NvidiaLLMClient(api_key=LLM_API_KEY, model=LLM_MODEL)
# from qwen_client import QwenLLMClient

# llm = QwenLLMClient()
NO_INFO_MSG = "No information found in the knowledge graph."

def generate_answer(question, results):
    if not results:
        return NO_INFO_MSG
    
    context_parts = []
    for r in results:
        if isinstance(r, dict):
            if 'entity' in r and 'related_entity' in r: 
                rel = r.get('relationship', 'is related to').replace("_", " ")
                context_parts.append(f"- {r['entity']} {rel} {r['related_entity']}")
            elif 'name' in r:
                context_parts.append(f"- {r['name']}")
        else:
            context_parts.append(f"- {str(r)}")
    
    if not context_parts:
        return NO_INFO_MSG

    context = "\n".join(context_parts[:25])
    
    # "TRY HARD" PROMPT: Combine facts with LLM knowledge
    prompt = f"""You are a climate science expert.

Knowledge Graph Facts:
{context}

Question: {question}

Instructions:
1. Answer the question by combining the Knowledge Graph Facts with your own expertise.
2. Be direct and concise (1-3 sentences).
3. DO NOT start your answer with phrases like "Based on the facts", "According to the graph", or "I couldn't find". Just give the answer directly.
"""
    
    try:
        response = llm.generate(
            user_prompt=prompt,
            system_prompt="You are a helpful and direct science assistant.",
            max_tokens=100
        ).strip()

        lower_res = response.lower()
        
        # 1. TRUE FAILURES: If it genuinely gave up
        if "i do not know" in lower_res or "i don't know" in lower_res:
            return NO_INFO_MSG
            
        # 2. CLEANING: Remove annoying conversational filler instead of failing the whole answer
        prefixes_to_remove = [
            "based on the information provided,",
            "based on the provided facts,",
            "based on the knowledge graph,",
            "according to the facts,",
            "in the knowledge graph,",
            "based on the information,"
        ]
        
        for prefix in prefixes_to_remove:
            if lower_res.startswith(prefix):
                # Slice off the prefix and strip whitespace
                response = response[len(prefix):].strip()
                lower_res = response.lower()
                
        # Capitalize the first letter if we sliced it
        if response:
            response = response[0].upper() + response[1:]

        return response
        
    except Exception:
        return NO_INFO_MSG