"""Assemble a carousel without altering the approved first image tensor."""
import torch
import torch.nn.functional as F

class KRASTV3PreserveFirstGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {name: ('IMAGE',) for name in ('first','second','third','fourth')}}
    RETURN_TYPES = ('IMAGE','IMAGE')
    RETURN_NAMES = ('grid','batch')
    FUNCTION = 'assemble'
    CATEGORY = 'KRASTv3 Carousel'

    def assemble(self, first, second, third, fourth):
        images = (first, second, third, fourth)
        for image in images:
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
                raise ValueError('Each input must be one ComfyUI IMAGE [1,H,W,C].')
            if min(image.shape[1:3]) < 1 or image.shape[3] != first.shape[3]:
                raise ValueError('Images must have positive dimensions and matching channel counts.')
        height,width = first.shape[1:3]
        panels = [first]
        for image in images[1:]:
            image = image.to(device=first.device, dtype=first.dtype)
            if image.shape[1:3] != first.shape[1:3]:
                image = F.interpolate(image.permute(0,3,1,2), size=(height,width), mode='bilinear', align_corners=False).permute(0,2,3,1)
            panels.append(image)
        top = torch.cat(panels[:2],dim=2)
        bottom = torch.cat(panels[2:],dim=2)
        return (torch.cat((top,bottom),dim=1), torch.cat(panels,dim=0))

NODE_CLASS_MAPPINGS = {'KRASTV3PreserveFirstGrid': KRASTV3PreserveFirstGrid}
NODE_DISPLAY_NAME_MAPPINGS = {'KRASTV3PreserveFirstGrid':'KRASTv3 | Preserve first frame + 3 variations'}
