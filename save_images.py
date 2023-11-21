import pandas as pd


def save_images_as_binary(image_name, image_binary):
    xl = pd.ExcelFile("images and excel files/image_binary.xlsx")
    df = xl.parse("Лист1")

    new_data = pd.DataFrame({"image_name": [image_name], "image_binary": [image_binary]})
    df = pd.concat([df, new_data], ignore_index=True)

    with pd.ExcelWriter("images and excel files/image_binary.xlsx", mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name="Лист1", index=False)
