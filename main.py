from agents import (
    search_db, for_agent, against_agent,
    questioner_agent, judge_agent
)

def run_second_opinion(question):
    
    print("\n" + "="*60)
    print("SECONDOPINION — DEBATE SYSTEM")
    print("="*60)
    print(f"\nQuestion: {question}\n")
    
    # Shared pool
    print("Searching knowledge base...")
    pool = search_db(question, n=5)
    print(f"Retrieved {len(pool)} relevant documents\n")
    
    # Round 1
    print("-"*40)
    print("ROUND 1 — FOR AGENT")
    print("-"*40)
    r1_for = for_agent(question, pool)
    print(r1_for)
    
    print("\n" + "-"*40)
    print("ROUND 1 — AGAINST AGENT")
    print("-"*40)
    r1_against = against_agent(question, pool)
    print(r1_against)
    
    print("\n" + "-"*40)
    print("QUESTIONER AGENT — BLIND SPOTS")
    print("-"*40)
    blind_spots, new_docs = questioner_agent(
        question, r1_for, r1_against
    )
    print(blind_spots)
    
    # Update pool
    updated_pool = pool + new_docs
    
    # Round 2
    print("\n" + "-"*40)
    print("ROUND 2 — FOR AGENT ON BLIND SPOTS")
    print("-"*40)
    r2_for = for_agent(blind_spots, updated_pool)
    print(r2_for)
    
    print("\n" + "-"*40)
    print("ROUND 2 — AGAINST AGENT ON BLIND SPOTS")
    print("-"*40)
    r2_against = against_agent(blind_spots, updated_pool)
    print(r2_against)
    
    print("\n" + "="*60)
    print("JUDGE AGENT — FINAL VERDICT")
    print("="*60)
    verdict = judge_agent(
        question, r1_for, r1_against,
        blind_spots, r2_for, r2_against,
        updated_pool
    )
    print(verdict)
    print("\n" + "="*60)
    
    return verdict

if __name__ == "__main__":
    question = input("Enter your decision question: ")
    run_second_opinion(question)