from dataset_multisplit import build_dataloader
# If you use torchvision augmentations:
# import torchvision.transforms as T
# tfm = T.Compose([T.Resize((256,256)), T.ToTensor()])  # Note: ToTensor normalizes pixel values to [0,1]

root = "/vast/users/guangyi.chen/causal_group/yunlong.deng/CausalVerse/image_dataset"

# Example 1: Load SCENE1, auto padding (if image sizes may differ)
loader, ds = build_dataloader(
    root=root,
    split="SCENE1",
    batch_size=16,
    num_workers=4,
    pad_images=False,            # If you have unified resolution, set to False
    image_transform=None,       # Or pass a torchvision transform (e.g., tfm above)
    check_files=False,          # Set to True to filter missing files during initialization
)

for images, metas in loader:
    # images -> model inputs
    # metas  -> model targets (supervision)
    print(images.shape, metas.shape)
    break

# Example 2: Load FALL, use your own transform, disable verbose
# loader, ds = build_dataloader(
#     root=root, split="FALL",
#     batch_size=32, pad_images=False,
#     image_transform=tfm, verbose=False
# )
