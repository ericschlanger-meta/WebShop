from pathlib import Path
from rich import print
from rich.markup import escape
from web_agent_site.envs import WebAgentSiteEnv
from web_agent_site.models import (
    QwenOutputPolicy,
    QwenPolicy,
)
from web_agent_site.utils import (
    setup_logger,
    DEBUG_PROD_SIZE,
)
import json
import logging
import requests

PARALLEL_ENVS = 1 
TOTAL_PRODUCTS = 1000

def is_bb_visible(bb, y_offset, viewport_height):
    """
    True if the element is fully visible in the viewport
    """
    bb_bottom = bb['y'] + bb['height'] 
    bb_top = bb['y']
    return bb_bottom <= viewport_height + y_offset and bb_top >= y_offset

def format_session(idx):
    return f'fixed_{idx}'

def call_llm(prompt_map):
    prompt_keys = list(prompt_map.keys())
    prompt_keys.sort()

    prompts = [prompt_map[k] for k in prompt_keys]

    post_response = requests.post(
        'https://q0xgtmmhf9hmeg-4000.proxy.runpod.net/predict_batch',
        json = {
            'token': 'ericsecret',
            'prompts': prompts,
        },
    )
    post_response_json = post_response.json()
    print(post_response_json)
    result = post_response_json['result']
    response_map = {}
    for idx, r in enumerate(result):
        response_map[prompt_keys[idx]] = r
    
    return response_map

def get_next_session_id(active_envs):
    sorted_keys = list(active_envs.keys())
    sorted_keys.sort()
    greatest_session_int = int(sorted_keys[-1].split('_')[1])
    incremented = greatest_session_int + 1
    return f'fixed_{incremented}'

if __name__ == '__main__':
    # Set up loggers
    # for i in range(TOTAL_PRODUCTS):
    for i in range(5):
        session_id = format_session(i)
        setup_logger(session_id, Path(f'user_session_logs/mturk/{session_id}'))

    # Initialize first N envs
    active_envs = {
        f'fixed_{i}': WebAgentSiteEnv(
            observation_mode='html',
            render=False,
            session=f'fixed_{i}',
        ) for i in range(PARALLEL_ENVS)
    }
    completed_envs = 0
    current_steps = {}
    instructions = {}
    observations = {}
    policies = {}
    resolutions = {}

    while completed_envs < TOTAL_PRODUCTS:
        # Gather available actions from all active envs + take screenshots
        available_actions = {}
        for env in active_envs.values():
            observations[env.session] = env.observation
            resolutions[env.session] = env.get_resolution()
            available_actions[env.session] = env.get_available_actions()
            instructions[env.session] = env.instruction_text
            if env.session not in policies.keys():
                policies[env.session] = QwenOutputPolicy()
            if env.session not in current_steps.keys():
                current_steps[env.session] = 1
            env.save_screenshot(current_steps[env.session])

        # Batch call LLM with active env prompts
        active_sessions = list(active_envs.keys())
        # Simplify this by moving get_prompt impl into stateful QwenOutputPolicy?
        prompts = {f'{session}': QwenPolicy.get_prompt(
            instructions[session],
            policies[session].previous_actions,
            observations[session],
            available_actions[session],
        ) for session in active_sessions}
        response_map = call_llm(prompts)

        # Parse actions into map from session_id to action
        driver_actions = {}
        bbs = {}
        for session in active_sessions:
            action = policies[session].forward(response_map[session])
            driver_actions[session] = action
            bbs[session] = active_envs[session].get_bb(action)

        # Scroll the screen for any envs that require scrolling
        for session in active_envs.keys():
            env = active_envs[session]
            logger = logging.getLogger(session)
            resolution = resolutions[session]
            bb = bbs[session]
            while (not is_bb_visible(bb, env.get_y_offset(), env.viewport_height)):
                scroll_distance = env.viewport_height / 2
                env.scroll_down(scroll_distance)
                logger.info(json.dumps(dict(
                    action=f"scroll[{scroll_distance}]",
                    resolution=resolution,
                )))
                current_steps[session] += 1
                env.save_screenshot(current_steps[session])
                bbs[session] = env.get_bb(driver_actions[session])
                bb = bbs[session]

        # Take the action
        completed_session_ids = []
        for session in active_envs.keys():
            action = driver_actions[session]
            env = active_envs[session]
            logger = logging.getLogger(session)
            logger.info(json.dumps(dict(
                action=action,
                bb=bbs[session],
                html=observations[session],
                resolution=resolutions[session],
            )))
            observations[session], reward, done, info = env.step(action)
            print(f'session {session} Taking action "{escape(action)}" -> Reward = {reward}')
            current_steps[session] += 1
            env.save_screenshot(current_steps[session])
            if done:
                logger.info(json.dumps(dict(
                    action='done',
                    reward=reward,
                    html=observations[session],
                    resolution=resolutions[session],
                )))
                env.close()
                completed_envs += 1
                completed_session_ids.append(session)

        for session in completed_session_ids:
            # Activate the next environment task
            next_session_id = get_next_session_id(active_envs)
            active_envs[next_session_id] = WebAgentSiteEnv(
                observation_mode='html',
                render=False,
                session=next_session_id,
            )
            # Clean up
            del active_envs[session]
            del current_steps[session]
            del instructions[session]
            del observations[session]
            del policies[session]
            del resolutions[session]

def old():
    log_dir = Path(f'user_session_logs/mturk/{session_id}')
    setup_logger(session_id, log_dir)
    env = WebAgentSiteEnv(
        observation_mode='html',
        run_id=pid,
        render=False,
        num_products=DEBUG_PROD_SIZE,
        session=session_id,
    )
    logger = logging.getLogger(session_id)

    try:
        policy = QwenPolicy(env.instruction_text)
        observation = env.observation
        env.save_screenshot(step+1)
        resolution = env.get_resolution()

        while True:
            available_actions = env.get_available_actions()
            action = policy.forward(observation, available_actions)
            bb = env.get_bb(action)
            while (not is_bb_visible(bb, env.get_y_offset(), env.viewport_height)):
                scroll_distance = env.viewport_height / 2
                env.scroll_down(scroll_distance)
                logger.info(json.dumps(dict(
                    action=f"scroll[{scroll_distance}]",
                    resolution=resolution,
                )))
                step += 1
                env.save_screenshot(step+1)
                bb = env.get_bb(action)

            logger.info(json.dumps(dict(
                action=action,
                bb=bb,
                html=observation,
                resolution=resolution,
            )))
            observation, reward, done, info = env.step(action)
            print(f'session {session_id} Taking action "{escape(action)}" -> Reward = {reward}')
            step += 1
            env.save_screenshot(step+1)
            if done:
                logger.info(json.dumps(dict(
                    action='done',
                    html=observation,
                    resolution=resolution,
                )))
                break
    finally:
        env.close()