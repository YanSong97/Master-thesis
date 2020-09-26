import torch
import time

"""
For LLB model
"""

time1 = time.time()
testing_reward_list = []
behaviour_uncertainty = True
deterministic = False
plan = 'rp'
num_data = 10

# mean_training_reward_list = []
# std_training_reward_list = []
torch.manual_seed(1)

agent = Agent(env_case=1, deterministic=deterministic, device='cuda', model='LLB')
for i in range(num_data):
    _ = agent.env_rollout(True, behaviour_uncertainty=behaviour_uncertainty, plan='random')

agent.model_learning(num_epoch=1000, num_batch=10)
# agent.model = BRNN_model

agent.policy_learning(imagine_num=50, num_particle=1000, num_epoch=1,
                      batch_size=10, horizon=10, plan=plan, w_uncertainty=True, e_uncertainty=True)

# mean_training_reward_list.append(mean_training_reward)
# std_training_reward_list.append(std_training_reward)

print('\n ------------------TESTING-------------------')
# over 10 trails
avg_rewards = 0
for j in range(20):
    rewards = agent.env_rollout(if_remember=False, behaviour_uncertainty=behaviour_uncertainty, plan=plan)
    print(j, rewards)
    avg_rewards += rewards
avg_rewards = avg_rewards / 20
testing_reward_list.append(avg_rewards)
print('Total trajs:', j, avg_rewards)
if avg_rewards >= 200:
    print('success')


#testing_reward_list = []
#mean_training_reward_list = []
#std_training_reward_list = []
#agent.policy = controller(1, 4, 'cuda').to('cuda')
avg_data_length_list = []

for i in range(50):
    print('Epoch = ',i+1)

    _ = agent.env_rollout(True, behaviour_uncertainty = behaviour_uncertainty,plan = plan)

    total = 0
    for i in range(len(agent.observations_list)):
        total += len(agent.observations_list[i])
    print('average training data length = ', total/len(agent.observations_list))
    avg_data_length_list.append(total/len(agent.observations_list))

    print('\n Begin model learning...')
    agent.model_learning(num_epoch =  1000, num_batch = 10)
    print('\n Finish model learning...')

    #agent.policy = controller(1, 4, 'cuda').to('cuda')
    agent.policy_learning(imagine_num=50, num_particle = 1000, num_epoch = 1,
                          batch_size = 10, horizon = 10, plan = plan, w_uncertainty = True, e_uncertainty = True )
    #mean_training_reward_list.append(mean_training_reward)
    #std_training_reward_list.append(std_training_reward)
    print('\n Finish policy learning...')

    print('\n ------------------TESTING-------------------')
    #over 10 trails
    avg_rewards = 0
    for j in range(20):
        rewards = agent.env_rollout(if_remember=False, behaviour_uncertainty = behaviour_uncertainty,plan = plan)
        print(j, rewards)
        avg_rewards += rewards
    avg_rewards = avg_rewards/20
    testing_reward_list.append(avg_rewards)
    print('Total trajs:', j, avg_rewards)
    if avg_rewards > 200:
        print('success')
time2 = time.time()
print(time2 - time1)