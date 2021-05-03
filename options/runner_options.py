import numpy as np
import torch
from copy import deepcopy
import random
from option_critic import OptionCriticFeatures
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from experience_replay import ReplayBuffer

from utils import to_tensor
from logger import Logger
import time
import os
from datetime import datetime


class Runner(object):
    def __init__(self, args):
        # Set the random seed during training and deterministic cudnn

        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

        # Check if we are running evaluation alone
        self.evaling_checkpoint = args.eval_checkpoint != ""
        # Create Logger
        self.logger = Logger(logdir=args.logdir,
                             run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

        # Load Env
        # self.env = gym.make('beaverkitchen-v0')
        self.env.set_args(self.env.Args(
            human_play=False,
            level=args.level,
            headless=args.render_train and not args.render_eval,
            render_slowly=False))

        self.env.seed(self.args.seed)

        self.num_t_steps = 0
        self.num_e_steps = 0
        self.epsilon = self.args.epsilon_start
        self.state = self.env.reset()
        self.num_states = len(self.state)
        self.num_actions = len(self.env.sample_action())

        # Create Model
        self.option_critic = OptionCriticFeatures(
            env_name=self.args.env,
            in_features=self.num_states,
            num_actions=self.num_actions,
            num_options=self.args.num_options,
            temperature=self.args.temp,
            eps_start=self.args.epsilon_start,
            eps_min=self.args.epsilon_min,
            eps_decay=self.args.epsilon_decay,
            eps_test=self.args.optimal_eps,
            device=device
        )

        # Create a prime network for more stable Q values
        self.option_critic_prime = deepcopy(self.option_critic)

        self.optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=self.args.learning_rate)
        torch.nn.utils.clip_grad_norm(self.option_critic.parameters(), self.args.clip_value)

        self.replay_buffer = ReplayBuffer(capacity=self.args.max_history, seed=self.args.seed)

    def run(self):
        self.n_episodes = self.args.n_episodes if not self.args.eval_checkpoint else self.args.n_eval_episodes

        for ep in range(self.n_episodes):
            if not self.args.eval_checkpoint:
                train_return = self.run_episode(ep, train=True)

                timestamp = str(datetime.now())
                print("[{}] Episode: {}, Train Return: {}".format(timestamp, ep, train_return))
                self.logger.log_return("reward/train_return", train_return, ep)

            if ep % self.args.eval_freq == 0:
                # Output and plot eval episode results
                eval_returns = []
                for _ in range(self.args.n_eval_samples):
                    eval_return = self.run_episode(ep, train=False)
                    eval_returns.append(eval_return)

                eval_return = np.array(eval_returns).mean()

                timestamp = str(datetime.now())
                print("[{}] Episode: {}, Eval Return: {}".format(timestamp, ep, eval_return))
                self.logger.log_return("reward/eval_return", eval_return, ep)

            if ep % self.args.checkpoint_freq == 0:
                if not os.path.exists(self.args.modeldir):
                    os.makedirs(self.args.modeldir)

                model_dir = os.path.join(self.args.modeldir, "episode_{:05d}".format(ep))
                self.option_critic.save(model_dir)

        print("Done running...")

    def run_episode(self, ep, train=False):
        option_lengths = {opt: [] for opt in range(self.args.num_options)}
        obs = self.env.reset()
        state = self.option_critic.get_state(to_tensor(obs))

        greedy_option = self.option_critic.greedy_option(state)
        current_option = 0

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0
        rewards = 0
        cum_reward = 0

        while not done and ep_steps < self.args.max_steps_ep:
            epsilon = self.epsilon if train else 0.0

            if train:
                self.num_t_steps += 1
            else:
                self.num_e_steps += 1

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(
                    self.args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action_idx, logp, entropy = self.option_critic.get_action(state, current_option)

            action = np.zeros(self.num_actions)
            action[int(action_idx)] = 1.0
            next_obs, reward, done, info = self.env.step(action)
            rewards += reward

            if train:
                self.replay_buffer.push(obs, current_option, reward, next_obs, done)

                old_state = state
                state = self.option_critic.get_state(to_tensor(next_obs))

                option_termination, greedy_option = self.option_critic.predict_option_termination(state, current_option)

                # Render domain
                if (self.args.render_train and train) or (self.args.render_eval and not train):
                    self.env.render()

                actor_loss, critic_loss = None, None
                if len(self.replay_buffer) > self.args.batch_size:
                    actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                                               reward, done, next_obs, self.option_critic, self.option_critic_prime,
                                               self.args)
                    loss = actor_loss

                    if ep % self.args.update_frequency == 0:
                        data_batch = self.replay_buffer.sample(self.args.batch_size)
                        critic_loss = critic_loss_fn(self.option_critic, self.option_critic_prime, data_batch,
                                                     self.args)
                        loss += critic_loss

                    self.optim.zero_grad()
                    torch.autograd.set_detect_anomaly(True)
                    loss.backward()
                    self.optim.step()

                    if ep % self.args.freeze_interval == 0:
                        self.option_critic_prime.load_state_dict(self.option_critic.state_dict())

                self.logger.log_return("train/cum_reward", cum_reward, self.num_t_steps)
                self.logger.log_data(self.num_t_steps, actor_loss, critic_loss, entropy.item(), self.epsilon)

            # update global steps etc
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            cum_reward += (self.args.gamma ** ep_steps) * reward
            self.epsilon = max(self.args.epsilon_min, self.epsilon * self.args.epsilon_decay)

            self.logger.log_return("error/volume_error_{}".format("train" if train else "eval"),
                                   float(info['volume_error']), self.num_t_steps if train else self.num_e_steps)

        self.logger.log_episode(self.num_t_steps, rewards, option_lengths, ep_steps, self.epsilon)

        return rewards

