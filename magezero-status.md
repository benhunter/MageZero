Last updated 12/22/25
## FAQ
* **-is there a python gym enviornment or API for AI in XMage?** no. This is something a lot of people want and I've brought it up with the XMage devs. Right now all AI projects I know of use their own fork of the game engine + a local server for the AI.  
* **-does this work with commander?**  yes, but not as well as 60 card constructed (takes longer to train)
* **-does this work with limited** yes because the hashed input space is shared, but since MageZero is deck-local; some additional tooling needs to be done to make a 'format-local' agent.
* **-can I play against the AI myself?** yes! see the setup guide for how.
* **-how long does training/self-play take?** full depth MCTS games take around 18 seconds on average with a mid/high end GPU and a 4.5ghz+ quad core cpu. Training times vary a lot by deck but generally 20K MCTS games is enough for most decks.
* **-does this use LLMs?** not right now, but the plan is to use them as a one time way to bootstrap RL. 
## Status
Over the last 9 months, I have been prototyping and testing the feasibility of a pure RL AlphaZero-style Monte Carlo Tree Search based deck-local agent in XMage. Despite optimized search for all decision points being an ongoing challenge in XMage. These early tests showed a lot of promise. So recently (last 3 months) I have been working to make this project more approachable for other interested developers since there has been a lot of interest in RL for MTG.

## Features
* RLvsMinimax gym environment (any 2 decks)
* RLvsRL gym environment (any 2 decks, any 2 models)
* RLvsHuman testing environment (any 2 decks)
* RL agent (MCTS) simulates out most micro-decisions in MTG (targeting, attacks/blocks, yes/nos, choosing modes) it also learns to predict opponent's plays indpendently.
* RL agent supports offline mode (which uses a heuristic value function and uniform priors) and online mode which uses a local server endpoint hosting a neural network for policy/value calculation. 
* gym environments can run games in parallel, however on most hardware going beyond 6 is not recomended due to XMage being very heap intensive. 
* gym environments are ran through JUnit in the XMage fork and exposes most hyper-params.
* pytorch project has some useful plotting functions (policy confusion matrix, value histogram, feature occurence plots, etc)

## Limitations
* **Deck-local** this is more of a design decision than a flaw, but there has been some confusion about this in the past. This system is *mostly* deck-local, meaning if you want the AI to play your deck well, you need to train it yourself. The one caveat being if your deck is similiar to an exisiting public archetype model, you can start with that but this is still an RL agent, not an MTG oracle. (see LLM goal)
* **Mana payment always uses autotapper** this is to keep branching factor from exploding too much. non-mana costs are still simulated out in tree search. If your deck has lands with complex activated abilities, the agent might not be able to fully learn them.
* **Training games are deterministic** while the state representation for the network excludes hidden info, the tree search is necessarily deterministic. meaning both agents play with information leakage about their future draws, opponent's hands, random outcomes etc. This is paritally mitigated through discount factors but remains a fundamental challenge of search in domains like MTG.
* **State blindness** some abilities like enter the dungeon and storm count arent encoded into the game state at the moment (see goals)
* **Missing some decisions** pre-game decisions (including mulligans), ordering triggers, and choosing replacement effects arent wired into MCTS yet.
* **Sideboarding/BO3 not supported** (see goals)
* **Multiplayer not supported** not a priority for me

## Goals

Short term
* **Train a suite of high quality mono colored standard power level agents** See the setup guide, chosen `.dck` files are in the project's XMage fork. this is probably the fastest, most helpful way to instantly get involved.
* **Continue to test and train more decks in XMage** Try different value functions, PUCT formulas, network architectures, state representations, hyper-parameters etc.
* **Expand feature vocabulary** Lots of niche mechanics are invisable to the agent right now. (enter the dungeon, phased out, pre-game decisions, ordering triggers). These could easily implemented in `StateEncoder.java` but haven't come up enough to justify implementation yet.
* **Implement mulligan decisions** Not used yet for AIvsAI simplicity, but will probably increase game signal for training so I will do this soon.
* **Optimize search** I've done all I can I fear; need help from experienced java/XMage engine devs at this point. 
* **Make async RL data pipeline** Right now the RL loop is `XMage self play -> hdf5 -> pytorch network -> local flask server -> XMage` this means hdf5 files need to be be manually moved after each generation of RL to the pytorch project for network training. This is good for incremental testing, but for production RL a true async replay buffer option would be much better. I made it so XMage writes the files live and chose hdf5 because it supports streaming and lazy loading to leave room for this upgrade down the line.
* **Support BO3** not a priority right now. but something that could easily (and should) be integrated into the RL process.
* **Collabortation** make/improve discord server. Make documentation. Community outreach in XMage and MTG dev community.


Long term
* **LLM integration** There have been a lot of debates about what role LLMs should play in MTG AI but personally I believe LLMs are best used as an oracle to kickstart a more specialized, scalable RL. I'm not exactly sure what the best way to do this would be, but it is definitely something I would love someone who knows more about LLMs than me to help with.(suggestions have been SoRA, RAG, SFT)
* **Publish Paper** MTG sits at the extremely relevant intersection between symbolic rules/languages and RL and I would love to be able to formally publish findings given the opportunity.
* **Hidden Info MCTS** Hidden info is a foundational challenge with MCTS. Advanced behaviors like belief modeling, bluffing, hedging etc. likely require another system/foundational change beyond the traditional Markov-Decision-Process of MTG.
* **MuZero-style dynamics model** very long-term research oriented goal I have. not only would a MuZero type model be able to avoid the need for in-simulator search, it could also solve the imperfect info problem. 
 
## High level
There is not one unified primary goal of this project since this was something I started for fun and mostly just wanted to make a working RL framework in XMage. But as an enjoyer of deckbuilding in MTG I would love for this project to evolve into a tool for deckbuilders and theorycrafters; think Untapped.gg or 17Lands but simulated live with your decks, against an evolving virtual metapool; allowing for personalized Card/WR/Matchup stats from 1000s of high-quality simulated games. Basically I want to give MTG an autobattler mode. 


While I am primarily focused on constructed right now. This system can easily be adapted to limited and EDH and pretty much any other format supported in XMage. My primary concern has just been improving the RL agent, and haven't really considered how this would be used until recently, so if you have an idea/project you want to use this for let me know :)

