// include the library code:
#include <LiquidCrystal.h>

//LiquidCrystal lcd(rs, en, d4, d5, d6, d7);
  LiquidCrystal lcd(13, 12, 11, 10,  9,  8);

const int L_Pin = 6;
const int M_Pin = 7;
const int Fan_Pin = 4;

void setup() 
{
  lcd.begin(20, 4); // set up the LCD's number of columns and rows:
  lcd.setCursor(0,0); // set the cursor position:
  lcd.print("    GREEN HOUSE    ");

  pinMode(L_Pin,OUTPUT);
  pinMode(M_Pin,OUTPUT);
  pinMode(Fan_Pin, OUTPUT);

}

void loop() 
{
  //Temperature Sensing 
  int S1=analogRead(A0);  // Read Temperature
  int Temp=(S1*500)/1023; // Storing value in Degree Celsius
  lcd.setCursor(0,1);
  lcd.print(" T=");
  lcd.print(Temp);
  lcd.print("'C     ");

  //Light Intensity
  int S2=analogRead(A1);  // Read Light Intensity
  int LI=S2/1.9;
  lcd.setCursor(11,1);
  lcd.print(" L=");
  lcd.print(LI);
  lcd.print("Lx       ");
  
  //Soil Moisture
  int S3=analogRead(A2);  // Read Soil Moisture 
  int SM=S3/10;
  lcd.setCursor(0,2);
  lcd.print(" S=");
  lcd.print(SM);
  lcd.print("%      ");

  //Air Humidity
  int S4=analogRead(A3);  // Read Air Humidity
  int H=S4/10;
  lcd.setCursor(11,2);
  lcd.print(" H=");
  lcd.print(H);
  lcd.print("%   ");
  
  if(LI<30)
  {
    digitalWrite(L_Pin,HIGH);
    lcd.setCursor(0,3);
    lcd.print(" Light:ON   ");
  }
  else
  {
    digitalWrite(L_Pin,LOW);
    lcd.setCursor(0,3);
    lcd.print(" Light:OFF   ");
  }

  if(SM<40)
  {
    digitalWrite(M_Pin,HIGH);
    lcd.setCursor(10,3);
    lcd.print(" Motor:ON    ");
  }  
  else
  {
    digitalWrite(M_Pin,LOW);
    lcd.setCursor(10,3);
    lcd.print(" Motor:OFF    ");
  }

  if (H<30)
  {
    digitalWrite(Fan_Pin,HIGH);
    
  }
  else
  {
    digitalWrite(Fan_Pin, LOW);
  }
}

