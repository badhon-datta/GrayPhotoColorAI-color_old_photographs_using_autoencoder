# Assets Folder

## Adding Your Demo Image

Place your before/after colorization demo image here as `demo.png`.

### How to create the demo image:

1. After training your model, pick a good example colorization
2. Create a side-by-side comparison showing:
   - Left side: Original grayscale image
   - Right side: AI-colorized result
3. Save it as `demo.png` in this folder

### Recommended size:
- Width: 1200-1600px
- Height: 400-600px
- Format: PNG or JPG

### Quick Python script to create side-by-side:

```python
from PIL import Image
import numpy as np

# Load your grayscale and colorized images
grayscale = Image.open("grayscale.jpg")
colorized = Image.open("colorized.jpg")

# Resize to same height
height = 400
grayscale = grayscale.resize((int(grayscale.width * height / grayscale.height), height))
colorized = colorized.resize((int(colorized.width * height / colorized.height), height))

# Create side-by-side
total_width = grayscale.width + colorized.width + 20  # 20px gap
combined = Image.new('RGB', (total_width, height), (255, 255, 255))
combined.paste(grayscale, (0, 0))
combined.paste(colorized, (grayscale.width + 20, 0))

# Save
combined.save("demo.png")
```

For now, the README will show a broken image icon until you add `demo.png` here.
