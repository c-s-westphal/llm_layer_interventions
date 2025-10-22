"""Diagnostic script to check available SAE IDs for gpt2-small-res-jb release."""

from sae_lens import SAE

print("=" * 80)
print("Checking SAE availability for gpt2-small-res-jb")
print("=" * 80)

# Test different hook point formats
test_cases = [
    "blocks.0.resid_post",
    "blocks.0.hook_resid_post",
    "blocks.0.resid_pre",
    "blocks.0.hook_resid_pre",
]

print("\nTesting different hook point formats:")
print("-" * 80)

for sae_id in test_cases:
    try:
        print(f"\nTrying: {sae_id}...")
        sae = SAE.from_pretrained(
            release='gpt2-small-res-jb',
            sae_id=sae_id,
            device='cpu'
        )
        print(f"✅ SUCCESS: '{sae_id}' works!")
        print(f"   SAE config: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
        break  # Found a working format
    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {sae_id}")

        # Try to extract valid IDs from error message
        if "Valid IDs" in error_msg or "available" in error_msg.lower():
            print(f"   Error hints at valid IDs:")
            # Print first 500 chars of error to see valid IDs
            print(f"   {error_msg[:500]}")
        else:
            print(f"   Error: {error_msg[:200]}")

print("\n" + "=" * 80)
print("Trying to list all available SAE IDs...")
print("=" * 80)

try:
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    directory = get_pretrained_saes_directory()

    release = 'gpt2-small-res-jb'
    if release in directory:
        print(f"\n✅ Found release: {release}")
        print(f"\nAvailable SAE IDs ({len(directory[release].saes_map)} total):")
        for i, sae_id in enumerate(directory[release].saes_map.keys()):
            print(f"  {i:2d}. {sae_id}")
            if i >= 14:  # Show first 15
                remaining = len(directory[release].saes_map) - 15
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break
    else:
        print(f"❌ Release '{release}' not found in directory")
        print(f"\nAvailable releases:")
        for i, rel in enumerate(list(directory.keys())[:20]):
            print(f"  {i:2d}. {rel}")

except ImportError as e:
    print(f"❌ Could not import pretrained_saes_directory: {e}")
except Exception as e:
    print(f"❌ Error listing SAEs: {e}")

print("\n" + "=" * 80)
print("Diagnosis complete!")
print("=" * 80)
