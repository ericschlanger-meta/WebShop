import gym
import random
import requests
import string
import os
import time

from bs4 import BeautifulSoup
from bs4.element import Comment
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotInteractableException, NoSuchElementException
from web_agent_site.engine.engine import parse_action, END_BUTTON

class WebAgentSiteEnv(gym.Env):
    """Gym environment for HTML mode of WebShop environment"""

    RESOLUTIONS = [
        {'width': 1920, 'height': 1080},
        {'width': 1536, 'height': 864},
        {'width': 1366, 'height': 768},
        {'width': 1280, 'height': 720},
    ]
    viewport_width = 0
    viewport_height = 0

    def __init__(self, observation_mode='html', **kwargs):
        """
        Constructor for HTML environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        pause (`float`) -- Pause (in seconds) after taking an action. 
            This is mainly for demo purposes.
            Recommended value: 2.0s
        render (`bool`) -- Show browser if set to `True`.
        session ('str') -- Session ID to initialize environment with
        """
        super(WebAgentSiteEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        # Create a browser driver to simulate the WebShop site
        options = Options()
        options.add_argument('--force-device-scale-factor=1')
        if 'render' not in kwargs or not kwargs['render']:
            options.add_argument("--headless")  # don't show browser
        self.browser = webdriver.Chrome(options=options)
        self.resolution = random.choice(self.RESOLUTIONS)
        self.viewport_width = self.resolution['width']
        self.viewport_height = self.resolution['height']

        # Set flags and values for WebShop session
        self.text_to_clickable = None
        self.radio_names_to_labels = {}
        self.assigned_session = kwargs.get('session')
        self.session = None
        self.reset()

    def get_resolution(self):
        return self.resolution

    def get_bb(self, action):
        """
        Returns the bounding box for the given action element
        """
        action_name, action_arg = parse_action(action)
        if action_name == 'search':
            try:
                search_bar = self.browser.find_element(By.ID, 'search_input')
            except NoSuchElementException:
                raise ValueError('no search bar found')
            return search_bar.rect
        elif action_name == 'click':
            button = self.text_to_clickable[action_arg]
            if action_arg in self.radio_names_to_labels.keys():
                return self.radio_names_to_labels[action_arg].rect
            else:
                return button.rect
        return None
    
    def get_y_offset(self):
        return self.browser.execute_script("return window.pageYOffset;")
    
    def scroll_down(self, amount):
        self.browser.execute_script(f"window.scrollBy(0, {amount});")

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info, bounding_box)
        If the element to be interacted with is outside below the viewport, this function will scroll the viewport
        and take a screenshot until the element is within the viewport.

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        reward = 0.0
        done = False
        info = None

        # Map action to executed command on the WebShop environment via the broswer driver
        action_name, action_arg = parse_action(action)
        if action_name == 'search':
            try:
                search_bar = self.browser.find_element(By.ID, 'search_input')
            except NoSuchElementException:
                print("No search bar found")
            else:
                search_bar.send_keys(action_arg)
                search_bar.submit()
        elif action_name == 'click':
            try:
                self.text_to_clickable[action_arg].click()
            except ElementNotInteractableException:
                # Perform force click with JavaScript
                button = self.text_to_clickable[action_arg]
                self.browser.execute_script("arguments[0].click();", button)
            reward = self.get_reward()
            if action_arg == END_BUTTON:
                done = True
        elif action_name == 'end':
            done = True
        else:
            print('Invalid action. No action performed.')

        if 'pause' in self.kwargs:
            time.sleep(self.kwargs['pause'])
        return self.observation, reward, done, info
    
    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        # Determine if a search bar is available
        try:
            search_bar = self.browser.find_element(By.ID, 'search_input')
        except NoSuchElementException:
            has_search_bar = False
        else:
            has_search_bar = True

        # Collect buttons, links, and options as clickables
        buttons = self.browser.find_elements(By.CLASS_NAME, 'btn')
        try:
            product_links = self.browser.find_elements(By.CLASS_NAME, 'product-link')
        except NoSuchElementException:
            product_links = []
        
        try:
            buying_options = self.browser.find_elements(By.CSS_SELECTOR, "input[type='radio']")
        except NoSuchElementException:
            buying_options = []

        self.text_to_clickable = {
            f'{b.text}': b
            for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get_attribute('value')
            self.text_to_clickable[f'{opt_value}'] = opt
            opt_id = opt.get_attribute('id')
            label = self.browser.find_element(By.CSS_SELECTOR, f"label[for='{opt_id}']")
            self.radio_names_to_labels[f'{opt_value}'] = label
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def _parse_html(self, html=None, url=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            if url is not None:
                html = requests.get(url)
            else:
                html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    def get_reward(self):
        """Get reward value at current step of the environment"""
        html_obj = self._parse_html()
        r = html_obj.find(id='reward')
        r = float(r.findChildren("pre")[0].string) if r is not None else 0.0
        return r
    
    def get_instruction_text(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text
    
    def convert_html_to_text(self, html):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        observation = ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        return observation
    
    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )
    
    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html)
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    @property
    def action_space(self):
        # Recommended to use `get_available_actions` instead
        return NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def save_screenshot(self, step_idx):
        user_log_dir = Path(f'user_session_logs/mturk/{self.session}')
        if not os.path.exists(user_log_dir):
            os.makedirs(user_log_dir)
        user_log_dir.mkdir(parents=True, exist_ok=True)
        self.browser.save_screenshot(Path.joinpath(user_log_dir, f'{step_idx}.png'))

    def reset(self):
        """Create a new session and reset environment variables"""
        if self.assigned_session is not None:
            self.session = self.assigned_session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        init_url = f'http://127.0.0.1:3000/{self.session}'
        self.browser.set_window_size(self.viewport_width, self.viewport_height)
        self.browser.get(init_url)

        self.instruction_text = self.get_instruction_text()

        return self.observation, None

    def render(self, mode='human'):
        # TODO: Render observation in terminal or WebShop website
        return NotImplementedError

    def close(self):
        # TODO: When DB used instead of JSONs, tear down DB here
        self.browser.close()
        print('Browser closed.')

def tag_visible(element):
    """Helper method to strip HTML block of extraneous tags"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )
