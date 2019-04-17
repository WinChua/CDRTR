make sentitrain DATA=Office_Products_Tools_and_Home_Improvement DOMAIN=Office_Products
make sentitrain DATA=Digital_Music_Toys_and_Games DOMAIN=Digital_Music
make sentitrain DATA=Office_Products_Tools_and_Home_Improvement DOMAIN=Tools_and_Home_Improvement
make sentitrain DATA=Digital_Music_Toys_and_Games DOMAIN=Toys_and_Games
make sentitrain DATA=Beauty_Clothin_Shoes_and_Jewelry DOMAIN=Jewelry
make sentitrain DATA=Cell_Phones_and_Accessories_Video_Games DOMAIN=Cell_Phones
make sentitrain DATA=Cell_Phones_and_Accessories_Video_Games DOMAIN=Video_Games
# make sentitrain DATA=Kindle_Movies_and_TV DOMAIN=Kindle
# make sentitrain DATA=CDs_and_Vinyl_Electronics DOMAIN=CDs_and_Vinyl
# make sentitrain DATA=CDs_and_Vinyl_Electronics DOMAIN=Electronics
# make sentitrain DATA=Kindle_Movies_and_TV DOMAIN=Movies_and_TV
## ./debug_sentitrain_Office_Products_3,5,7,11_100.log:[D 4.269944
## ./debug_sentitrain_Digital_Music_3,5,7,11_100.log:[D 5.784357
## ./debug_sentitrain_Tools_and_Home_Improvement_3,5,7,11_100.log:[D 6.728158
## ./debug_sentitrain_Toys_and_Games_3,5,7,11_100.log:[D 7.636538
## ./debug_sentitrain_Jewelry_3,5,7,11_100.log:[D 8.161653
## ./debug_sentitrain_Cell_Phones_3,5,7,11_100.log:[D 9.358803
## ./debug_sentitrain_Video_Games_3,5,7,11_100.log:[D 17.118057
## ./debug_sentitrain_Kindle_3,5,7,11_100.log:[D 52.382837
## ./debug_sentitrain_CDs_and_Vinyl_3,5,7,11_100.log:[D 78.811448
## ./debug_sentitrain_Electronics_3,5,7,11_100.log:[D 95.606552
## ./debug_sentitrain_Movies_and_TV_3,5,7,11_100.log:[D 113.493708
## find . -mmin -40 -maxdepth 1 -type f | xargs grep cost | cut -d' ' -f1,11 | sort -k2 -h >> ../senti.sh
