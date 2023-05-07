import torch
from diffusers import StableDiffusionPipeline
from . import AutoGPTPluginTemplate

class AutoGPTMakeImagePlugin(AutoGPTPluginTemplate):

    """This plugin is used to make an image with Stable Diffusion."""

    def __init__(self):
        """The constructor method."""

        super().__init__()
        self._name = "Auto-GPT-Image-Maker"
        self._version = "0.1.0"
        self._description = "This plugin is used to make an image with Stable Diffusion."

    # End of __init__ method.


    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.

        Returns:
            bool: True if the plugin can handle the post_prompt method."""

        return True
    
    # End of can_handle_post_prompt method.
    
    
    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.
        Args:
            prompt (PromptGenerator): The prompt generator.
        Returns:
            PromptGenerator: The prompt generator.
        """
        prompt.add_command(
            "make_image", 
            "Make an Image",
            self.make_image
        )

    # End of post_prompt method.


    def make_image(self, prompt: str):
        """This method is called when the make_image command is called."""

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        image = pipe(prompt).images[0]  
    
        image.save("astronaut_rides_horse.png")

    # End of make_image method.