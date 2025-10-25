#!/usr/bin/env python3
"""
Diverse Name Inventory System for NER Training Augmentation

This module generates and manages a large, culturally diverse name inventory
to break name-role bias patterns in training data and improve model generalization.

Key Features:
- 10,000+ diverse names across multiple ethnicities
- Weighted generation with cultural consistency rules
- Gender-aware title selection
- Anti-bias mechanisms to prevent role memorization
- Fast lookup and sampling operations

Address the core issues identified:
- Catastrophic name repetition (e.g., "Wounded Hawk" used 723 times)
- Severe role bias (e.g., "Aria Nguyen" = 91.8% PROTAGONIST)
- Poor ethnic diversity (79.5% classified as "Other/Mixed")
"""

import random
import json
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NameInventory:
    """
    Comprehensive name inventory system for generating diverse, culturally
    authentic names for NER training data augmentation.

    Implements weighted sampling strategies to ensure:
    1. Cultural authenticity and consistency
    2. Gender-appropriate title selection
    3. Anti-bias mechanisms
    4. Massive name diversity (10,000+ unique combinations)
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the name inventory system."""
        if seed is not None:
            random.seed(seed)

        # Core data structures
        self.titles = {
            'gendered': {
                'male': ['Mr.', 'Sir'],
                'female': ['Ms.', 'Mrs.', 'Miss'],
                'neutral': ['Mx.']
            },
            'professional': [
                'Dr.', 'Professor', 'Captain', 'Detective', 'Sergeant',
                'Agent', 'Officer', 'Commander', 'Director', 'Chief',
                'Judge', 'Admiral', 'General', 'Colonel', 'Major',
                'Lieutenant', 'Minister', 'Ambassador', 'Chancellor'
            ]
        }

        # Comprehensive name databases by culture and gender
        self.first_names = {
            'Anglo/Western': {
                'male': [
                    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph',
                    'Thomas', 'Christopher', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark',
                    'Donald', 'Steven', 'Kenneth', 'Andrew', 'Brian', 'Joshua', 'Kevin',
                    'Edward', 'Ronald', 'Timothy', 'Jason', 'Jeffrey', 'Ryan', 'Jacob',
                    'Gary', 'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin',
                    'Scott', 'Brandon', 'Benjamin', 'Samuel', 'Gregory', 'Alexander',
                    'Nathan', 'Henry', 'Jack', 'Oliver', 'Harry', 'George', 'Oscar',
                    'Charlie', 'Leo', 'Noah', 'Arthur', 'Muhammad', 'Theo', 'Luke',
                    'Ethan', 'Lewis', 'Mason', 'Jacob', 'Logan', 'Alexander', 'Sebastian',
                    'Elijah', 'Isaiah', 'Gabriel', 'Carter', 'Adrian', 'Caleb', 'Wyatt',
                    'Connor', 'Eli', 'Hunter', 'Christian', 'Colton', 'Dominic', 'Blake'
                ],
                'female': [
                    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
                    'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Helen',
                    'Sandra', 'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle', 'Laura',
                    'Sarah', 'Kimberly', 'Deborah', 'Dorothy', 'Lisa', 'Nancy', 'Karen',
                    'Emily', 'Emma', 'Madison', 'Olivia', 'Hannah', 'Abigail', 'Isabella',
                    'Samantha', 'Elizabeth', 'Ashley', 'Alexis', 'Sarah', 'Sophia',
                    'Alyssa', 'Grace', 'Ava', 'Taylor', 'Brianna', 'Lauren', 'Chloe',
                    'Natalie', 'Kayla', 'Jessica', 'Anna', 'Victoria', 'Mia', 'Hailey',
                    'Sydney', 'Jordan', 'Destiny', 'Morgan', 'Rachel', 'Julia', 'Jasmine',
                    'Katherine', 'Brooke', 'Allison', 'Savannah', 'Andrea', 'Jenna',
                    'Caroline', 'Aria', 'Catherine', 'Violet', 'Scarlett', 'Cora'
                ]
            },
            'Hispanic/Latino': {
                'male': [
                    'José', 'Luis', 'Juan', 'Miguel', 'Carlos', 'Antonio', 'Francisco',
                    'Manuel', 'Alejandro', 'Pedro', 'Rafael', 'Daniel', 'Jorge', 'David',
                    'Jesús', 'Fernando', 'Roberto', 'Ricardo', 'Eduardo', 'Ramón',
                    'Sergio', 'Andrés', 'Javier', 'Diego', 'Marco', 'César', 'Salvador',
                    'Víctor', 'Hugo', 'Raúl', 'Óscar', 'Ernesto', 'Gabriel', 'Rodrigo',
                    'Arturo', 'Rubén', 'Armando', 'Enrique', 'Ignacio', 'Adrián',
                    'Emilio', 'Julio', 'Guillermo', 'Gonzalo', 'Iván', 'Mauricio',
                    'Felipe', 'Santiago', 'Sebastián', 'Nicolás', 'Mateo', 'Leonardo',
                    'Lucas', 'Thiago', 'Benjamín', 'Agustín', 'Joaquín', 'Bruno',
                    'Santino', 'Lorenzo', 'Gael', 'Ian', 'Enzo', 'Emiliano'
                ],
                'female': [
                    'María', 'Ana', 'Carmen', 'Josefa', 'Isabel', 'Dolores', 'Pilar',
                    'Teresa', 'Rosa', 'Francisco', 'Antonia', 'Esperanza', 'Mercedes',
                    'Elena', 'Concepción', 'Manuela', 'Cristina', 'Laura', 'Marta',
                    'Silvia', 'Sara', 'Paula', 'Beatriz', 'Raquel', 'Irene', 'Patricia',
                    'Rocío', 'Nuria', 'Andrea', 'Mónica', 'Sonia', 'Susana', 'Eva',
                    'Alicia', 'Verónica', 'Julia', 'Amparo', 'Inmaculada', 'Yolanda',
                    'Milagros', 'Encarnación', 'Ángeles', 'Montserrat', 'Victoria',
                    'Guadalupe', 'Lucía', 'Valentina', 'Isabella', 'Camila', 'Valeria',
                    'Mariana', 'Gabriela', 'Daniela', 'Victoria', 'Martina', 'Emma',
                    'Sofía', 'Regina', 'Renata', 'Amanda', 'Melissa', 'Nicole'
                ]
            },
            'East Asian': {
                'male': [
                    'Wei', 'Jun', 'Hiroshi', 'Akira', 'Takeshi', 'Kenji', 'Yuki', 'Satoshi',
                    'Kazuki', 'Ryota', 'Haruto', 'Yuto', 'Sota', 'Daiki', 'Kenta',
                    'Hayato', 'Ren', 'Jin', 'Ming', 'Hao', 'Feng', 'Gang', 'Qiang',
                    'Jie', 'Kai', 'Long', 'Peng', 'Bin', 'Tao', 'Yang', 'Cheng',
                    'Hyun', 'Min', 'Seung', 'Jong', 'Kyung', 'Sung', 'Joon', 'Woo',
                    'Ho', 'Dong', 'Yoon', 'Soo', 'Jin', 'Bum', 'Hoon', 'Young',
                    'Duc', 'Minh', 'Anh', 'Hung', 'Quan', 'Dung', 'Tuan', 'Long',
                    'Nam', 'Hieu', 'Khoa', 'Phong', 'Trong', 'Thanh', 'Cuong',
                    'Hiroto', 'Yamato', 'Riku', 'Souta', 'Kaito', 'Asahi', 'Hinata'
                ],
                'female': [
                    'Yuki', 'Akiko', 'Hiroko', 'Keiko', 'Michiko', 'Naoko', 'Tomoko',
                    'Sachiko', 'Masako', 'Kazuko', 'Haruka', 'Yui', 'Rina', 'Ai',
                    'Mei', 'Ling', 'Yu', 'Xin', 'Li', 'Min', 'Jing', 'Yan', 'Hong',
                    'Fang', 'Hui', 'Juan', 'Ping', 'Qing', 'Rui', 'Shan', 'Ting',
                    'Jin', 'Hye', 'Soo', 'Young', 'Mi', 'Kyung', 'Sun', 'Jung',
                    'Hee', 'Eun', 'Ji', 'Na', 'So', 'Yeon', 'In', 'Ah', 'Bin',
                    'Linh', 'Thu', 'Huong', 'Lan', 'Mai', 'Nga', 'Phuong', 'Trang',
                    'Yen', 'Thao', 'Van', 'Ha', 'Hoa', 'Kim', 'Le', 'My',
                    'Sakura', 'Hana', 'Kira', 'Mika', 'Nana', 'Rika', 'Saki'
                ]
            },
            'Middle Eastern/Arabic': {
                'male': [
                    'Ahmed', 'Mohammed', 'Ali', 'Omar', 'Hassan', 'Khalid', 'Saeed',
                    'Abdullah', 'Ibrahim', 'Yousef', 'Mansour', 'Salem', 'Faisal',
                    'Nasser', 'Waleed', 'Tariq', 'Majid', 'Saud', 'Fahad', 'Nawaf',
                    'Abdulaziz', 'Abdulrahman', 'Bandar', 'Turki', 'Mishari', 'Rayan',
                    'Zaid', 'Hamza', 'Osama', 'Adel', 'Badr', 'Sami', 'Marwan',
                    'Amjad', 'Mazen', 'Hisham', 'Fares', 'Nabil', 'Jamal', 'Karim',
                    'Rami', 'Amir', 'Yasser', 'Bassam', 'Tamer', 'Wael', 'Ashraf',
                    'Mahmoud', 'Mustafa', 'Bilal', 'Anas', 'Yazan', 'Majd', 'Laith',
                    'Qasim', 'Rashid', 'Salim', 'Nazir', 'Hakim', 'Jalal', 'Sharif'
                ],
                'female': [
                    'Fatima', 'Aisha', 'Khadija', 'Maryam', 'Zainab', 'Amina', 'Sarah',
                    'Layla', 'Nour', 'Hala', 'Rana', 'Dina', 'Reem', 'Lina', 'Nada',
                    'Maya', 'Yasmina', 'Salma', 'Rania', 'Leila', 'Amira', 'Nadine',
                    'Dalal', 'Maha', 'Widad', 'Siham', 'Rajaa', 'Zineb', 'Aicha',
                    'Samira', 'Malika', 'Halima', 'Kenza', 'Safaa', 'Hanae', 'Imane',
                    'Sanaa', 'Houda', 'Btissam', 'Karima', 'Zohra', 'Hayat', 'Nawal',
                    'Asma', 'Latifa', 'Jamila', 'Soraya', 'Farida', 'Najat', 'Ghita',
                    'Houria', 'Mina', 'Yasmine', 'Rim', 'Salam', 'Jana', 'Lara',
                    'Tala', 'Naya', 'Jude', 'Layan', 'Alma', 'Celine', 'Ghada'
                ]
            },
            'African': {
                'male': [
                    'Kwame', 'Kofi', 'Kojo', 'Yaw', 'Akwasi', 'Kwaku', 'Kwabena',
                    'Adwoa', 'Ama', 'Akosua', 'Yaa', 'Efua', 'Aba', 'Benewaa',
                    'Sekou', 'Ousmane', 'Moussa', 'Ibrahima', 'Amadou', 'Mamadou',
                    'Abdoulaye', 'Boubacar', 'Souleymane', 'Cheikh', 'Modou', 'Lamine',
                    'Chinedu', 'Chibueze', 'Emeka', 'Ikechukwu', 'Kelechi', 'Obinna',
                    'Onyeka', 'Ugochukwu', 'Nnamdi', 'Chukwudi', 'Chinonso', 'Chigozie',
                    'Sipho', 'Thabo', 'Lucky', 'Mandla', 'Sizani', 'Bheki', 'Muzi',
                    'Nkosinathi', 'Sandile', 'Mthunzi', 'Jabulani', 'Sibusiso',
                    'Tendai', 'Tinashe', 'Tafadzwa', 'Tapiwa', 'Takudzwa', 'Tatenda',
                    'Femi', 'Segun', 'Tunde', 'Kemi', 'Dele', 'Wale', 'Bayo'
                ],
                'female': [
                    'Akosua', 'Adwoa', 'Afia', 'Akua', 'Yaa', 'Ama', 'Abena',
                    'Efua', 'Esi', 'Aba', 'Adjoa', 'Araba', 'Benewaa', 'Comfort',
                    'Ama', 'Fatou', 'Aminata', 'Mariama', 'Aissatou', 'Khady',
                    'Ndeye', 'Bineta', 'Coumba', 'Rokhaya', 'Astou', 'Dieynaba',
                    'Chinelo', 'Chioma', 'Ngozi', 'Adaeze', 'Ifeoma', 'Nneka',
                    'Chiamaka', 'Chidinma', 'Uzoma', 'Amarachi', 'Chinyere', 'Obioma',
                    'Nomsa', 'Thandiwe', 'Nomzamo', 'Zanele', 'Busisiwe', 'Nosipho',
                    'Naledi', 'Mamello', 'Palesa', 'Refilwe', 'Tebogo', 'Boitumelo',
                    'Chipo', 'Rumbidzai', 'Muchaneta', 'Chiedza', 'Nyasha', 'Tsitsi',
                    'Folake', 'Funmi', 'Kemi', 'Ronke', 'Yemi', 'Bisi', 'Toyin'
                ]
            },
            'European': {
                'male': [
                    'Jean', 'Pierre', 'Michel', 'Alain', 'Philippe', 'André', 'Bernard',
                    'Daniel', 'Claude', 'Gérard', 'Patrick', 'François', 'Henri',
                    'Hans', 'Klaus', 'Jürgen', 'Peter', 'Wolfgang', 'Helmut', 'Heinz',
                    'Gerhard', 'Horst', 'Dieter', 'Werner', 'Günther', 'Frank',
                    'Giuseppe', 'Antonio', 'Mario', 'Franco', 'Bruno', 'Paolo',
                    'Giorgio', 'Marco', 'Roberto', 'Andrea', 'Stefano', 'Angelo',
                    'Lars', 'Erik', 'Ole', 'Per', 'Jan', 'Anders', 'Nils', 'Magnus',
                    'Karl', 'Johan', 'Mikael', 'Stefan', 'Mattias', 'Daniel',
                    'Andrei', 'Sergei', 'Vladimir', 'Dmitri', 'Alexander', 'Nikolai',
                    'Pavel', 'Mikhail', 'Alexei', 'Igor', 'Viktor', 'Oleg'
                ],
                'female': [
                    'Marie', 'Françoise', 'Monique', 'Catherine', 'Nathalie', 'Isabelle',
                    'Sylvie', 'Martine', 'Nicole', 'Chantal', 'Christine', 'Brigitte',
                    'Ursula', 'Ingrid', 'Petra', 'Monika', 'Gabriele', 'Sabine',
                    'Claudia', 'Birgit', 'Andrea', 'Karin', 'Martina', 'Susanne',
                    'Maria', 'Anna', 'Giuseppina', 'Rosa', 'Angela', 'Giovanna',
                    'Teresa', 'Lucia', 'Carmela', 'Caterina', 'Francesca', 'Paola',
                    'Ingrid', 'Astrid', 'Karin', 'Eva', 'Birgitta', 'Margareta',
                    'Elisabeth', 'Christina', 'Marianne', 'Gunilla', 'Monica', 'Anita',
                    'Olga', 'Elena', 'Natasha', 'Irina', 'Svetlana', 'Marina',
                    'Tatiana', 'Galina', 'Lyudmila', 'Valentina', 'Nina', 'Anna'
                ]
            },
            'South Asian': {
                'male': [
                    'Raj', 'Amit', 'Suresh', 'Ravi', 'Anil', 'Ashok', 'Vinod', 'Manoj',
                    'Sanjay', 'Ajay', 'Prakash', 'Deepak', 'Ramesh', 'Mukesh', 'Rajesh',
                    'Rohit', 'Nitin', 'Sachin', 'Gaurav', 'Akash', 'Arjun', 'Karan',
                    'Varun', 'Shivam', 'Ankit', 'Rishabh', 'Aditya', 'Kartik', 'Harsh',
                    'Kabir', 'Aryan', 'Aarav', 'Vihaan', 'Aayush', 'Sai', 'Dhruv',
                    'Krishna', 'Aryan', 'Ishaan', 'Shivansh', 'Atharv', 'Reyansh',
                    'Hassan', 'Ali', 'Ahmed', 'Usman', 'Bilal', 'Hamza', 'Zain',
                    'Umer', 'Hasan', 'Ibrahim', 'Saad', 'Osama', 'Omar', 'Sami',
                    'Kamal', 'Sahan', 'Nimal', 'Pradeep', 'Chaminda', 'Lasith',
                    'Mahela', 'Tillakaratne', 'Sanath', 'Aravinda', 'Marvan', 'Kumar'
                ],
                'female': [
                    'Priya', 'Sunita', 'Sita', 'Geeta', 'Anita', 'Rita', 'Lata',
                    'Usha', 'Asha', 'Kavita', 'Meera', 'Rekha', 'Shanti', 'Kamala',
                    'Pooja', 'Neha', 'Preeti', 'Deepika', 'Ritu', 'Sangeeta', 'Smita',
                    'Anjali', 'Sonal', 'Rashmi', 'Swati', 'Vandana', 'Kiran', 'Nisha',
                    'Aadhya', 'Saanvi', 'Aanya', 'Diya', 'Pihu', 'Prisha', 'Anaya',
                    'Fatima', 'Ayesha', 'Sana', 'Zara', 'Hira', 'Nimra', 'Eman',
                    'Alisha', 'Mahnoor', 'Zoya', 'Laiba', 'Hoorain', 'Rida', 'Inaya',
                    'Amal', 'Chathurika', 'Dilani', 'Hasini', 'Kavitha', 'Malsha',
                    'Nadeeka', 'Priyanka', 'Sanduni', 'Tharanga', 'Udari', 'Vindya'
                ]
            },
            'Indigenous': {
                'male': [
                    'Aiyana', 'Akecheta', 'Chayton', 'Dakota', 'Enapay', 'Hanska',
                    'Hototo', 'Keme', 'Kitchi', 'Kohana', 'Kuruk', 'Makoons',
                    'Mingan', 'Naalnish', 'Odakota', 'Pachu', 'Sahale', 'Takoda',
                    'Tuari', 'Waban', 'Yaholo', 'Aiyana', 'Aponi', 'Chenoa',
                    'Dyani', 'Halona', 'Isi', 'Kachina', 'Kaya', 'Leotie',
                    'Malia', 'Nayeli', 'Orenda', 'Pocahontas', 'Sacagawea', 'Tallulah',
                    'Uma', 'Winona', 'Yamka', 'Zaltana', 'Naira', 'Itzel', 'Yaretzi',
                    'Ximena', 'Citlali', 'Xochitl', 'Tlalli', 'Nelli', 'Itzi',
                    'Amoxtli', 'Cipactli', 'Ehecatl', 'Itzel', 'Malinali', 'Necalli',
                    'Quetzal', 'Teoxihuitl', 'Tlacaelel', 'Xochitl', 'Yaotl', 'Zyanya'
                ],
                'female': [
                    'Aiyana', 'Aponi', 'Chenoa', 'Dyani', 'Halona', 'Isi', 'Kachina',
                    'Kaya', 'Leotie', 'Malia', 'Nayeli', 'Orenda', 'Pocahontas',
                    'Sacagawea', 'Tallulah', 'Uma', 'Winona', 'Yamka', 'Zaltana',
                    'Naira', 'Itzel', 'Yaretzi', 'Ximena', 'Citlali', 'Xochitl',
                    'Tlalli', 'Nelli', 'Itzi', 'Amoxtli', 'Ehecatl', 'Malinali',
                    'Quetzal', 'Teoxihuitl', 'Xochitl', 'Yaotl', 'Zyanya', 'Paloma',
                    'Lua', 'Aiyana', 'Akira', 'Amara', 'Anyu', 'Atepa', 'Awentia',
                    'Ayita', 'Chitsa', 'Dena', 'Ehawee', 'Enola', 'Etaney', 'Halona',
                    'Hateya', 'Huyana', 'Istas', 'Ituha', 'Kachina', 'Kamama',
                    'Kaya', 'Kimama', 'Kinta', 'Litonya', 'Malia', 'Mika', 'Nahimana'
                ]
            },
            'Mixed/International': {
                'male': [
                    'Alex', 'Sam', 'Jordan', 'Casey', 'Taylor', 'Morgan', 'Jamie',
                    'Riley', 'Avery', 'Cameron', 'Quinn', 'Sage', 'River', 'Phoenix',
                    'Rowan', 'Kai', 'Ari', 'Emery', 'Reese', 'Blake', 'Lane',
                    'Peyton', 'Skyler', 'Dakota', 'Eden', 'Finley', 'Hayden',
                    'Harper', 'Lennon', 'Marlowe', 'Parker', 'Remy', 'Shiloh',
                    'Tate', 'Vale', 'Winter', 'Nova', 'River', 'Storm', 'Ocean',
                    'Atlas', 'Cruz', 'Dante', 'Felix', 'Leo', 'Max', 'Nico',
                    'Orion', 'Rio', 'Zoe', 'Aria', 'Luna', 'Mila', 'Noa',
                    'Isla', 'Ivy', 'Jade', 'Leah', 'Maya', 'Nina', 'Zara'
                ],
                'female': [
                    'Alex', 'Sam', 'Jordan', 'Casey', 'Taylor', 'Morgan', 'Jamie',
                    'Riley', 'Avery', 'Cameron', 'Quinn', 'Sage', 'River', 'Phoenix',
                    'Rowan', 'Kai', 'Ari', 'Emery', 'Reese', 'Blake', 'Lane',
                    'Peyton', 'Skyler', 'Dakota', 'Eden', 'Finley', 'Hayden',
                    'Harper', 'Lennon', 'Marlowe', 'Parker', 'Remy', 'Shiloh',
                    'Tate', 'Vale', 'Winter', 'Nova', 'River', 'Storm', 'Ocean',
                    'Atlas', 'Cruz', 'Dante', 'Felix', 'Leo', 'Max', 'Nico',
                    'Orion', 'Rio', 'Zoe', 'Aria', 'Luna', 'Mila', 'Noa',
                    'Isla', 'Ivy', 'Jade', 'Leah', 'Maya', 'Nina', 'Zara'
                ]
            }
        }

        # Comprehensive surname databases by culture
        self.surnames = {
            'Anglo/Western': [
                'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
                'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King',
                'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green',
                'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
                'Carter', 'Roberts', 'Cooper', 'Reed', 'Bailey', 'Bell', 'Gomez',
                'Kelly', 'Howard', 'Ward', 'Cox', 'Diaz', 'Richardson', 'Wood',
                'Watson', 'Brooks', 'Bennett', 'Gray', 'James', 'Reyes', 'Cruz',
                'Hughes', 'Price', 'Myers', 'Long', 'Foster', 'Sanders', 'Ross',
                'Morales', 'Powell', 'Sullivan', 'Russell', 'Ortiz', 'Jenkins',
                'Gutierrez', 'Perry', 'Butler', 'Barnes', 'Fisher', 'Henderson'
            ],
            'Hispanic/Latino': [
                'García', 'Rodríguez', 'González', 'Fernández', 'López', 'Martínez',
                'Sánchez', 'Pérez', 'Gómez', 'Martín', 'Jiménez', 'Ruiz', 'Hernández',
                'Díaz', 'Moreno', 'Muñoz', 'Álvarez', 'Romero', 'Alonso', 'Gutiérrez',
                'Navarro', 'Torres', 'Domínguez', 'Vázquez', 'Ramos', 'Gil', 'Ramírez',
                'Serrano', 'Blanco', 'Suárez', 'Molina', 'Morales', 'Ortega', 'Delgado',
                'Castro', 'Ortiz', 'Rubio', 'Marín', 'Sanz', 'Iglesias', 'Medina',
                'Garrido', 'Cortés', 'Castillo', 'Santos', 'Lozano', 'Guerrero',
                'Cano', 'Prieto', 'Méndez', 'Cruz', 'Flores', 'Herrera', 'Aguilar',
                'Vega', 'Mendoza', 'Vargas', 'Reyes', 'Rojas', 'Contreras', 'Silva',
                'Campos', 'Figueroa', 'Espinoza', 'Paredes', 'Ríos', 'Peña', 'Soto',
                'Cabrera', 'Fuentes', 'Santiago', 'Miranda', 'Sandoval', 'Salazar',
                'Rivas', 'Guerrero', 'Maldonado', 'Valdez', 'Esquivel', 'Pacheco'
            ],
            'East Asian': [
                'Wang', 'Li', 'Zhang', 'Liu', 'Chen', 'Yang', 'Huang', 'Zhao',
                'Wu', 'Zhou', 'Xu', 'Sun', 'Ma', 'Zhu', 'Hu', 'Guo', 'He', 'Lin',
                'Luo', 'Zheng', 'Liang', 'Song', 'Tang', 'Xu', 'Han', 'Feng',
                'Deng', 'Cao', 'Peng', 'Zeng', 'Xiao', 'Tian', 'Dong', 'Pan',
                'Sato', 'Suzuki', 'Takahashi', 'Tanaka', 'Watanabe', 'Ito', 'Yamamoto',
                'Nakamura', 'Kobayashi', 'Kato', 'Yoshida', 'Yamada', 'Sasaki',
                'Yamaguchi', 'Saito', 'Matsumoto', 'Inoue', 'Kimura', 'Hayashi',
                'Shimizu', 'Yamazaki', 'Mori', 'Abe', 'Ikeda', 'Hashimoto', 'Yamashita',
                'Ishikawa', 'Nakajima', 'Maeda', 'Fujita', 'Ogawa', 'Goto', 'Okada',
                'Kim', 'Lee', 'Park', 'Choi', 'Jung', 'Kang', 'Cho', 'Yoon',
                'Jang', 'Lim', 'Han', 'Oh', 'Seo', 'Shin', 'Kwon', 'Hwang',
                'Ahn', 'Kim', 'Song', 'Hong', 'Min', 'Moon', 'Bae', 'Yoo'
            ],
            'Middle Eastern/Arabic': [
                'Al-Ahmad', 'Al-Ali', 'Al-Hassan', 'Al-Hussein', 'Al-Mahmoud', 'Al-Rashid',
                'Al-Sabah', 'Al-Thani', 'Al-Zahra', 'Abdel-Rahman', 'Abu-Baker', 'Abu-Hassan',
                'Abu-Omar', 'Farouk', 'Habib', 'Mansour', 'Nasser', 'Omar', 'Qasemi',
                'Rashid', 'Saleh', 'Youssef', 'Zayed', 'Benali', 'Benaissa', 'Benzema',
                'Boudiaf', 'Cherif', 'Djebbar', 'Hadj', 'Hamdi', 'Khelifi', 'Laroui',
                'Mahfouz', 'Nadir', 'Ouali', 'Rezki', 'Saadi', 'Taleb', 'Zahra',
                'Amiri', 'Hosseini', 'Karimi', 'Mohammadi', 'Rahmani', 'Rostami',
                'Salehi', 'Ahmadi', 'Moradi', 'Kazemi', 'Sadeghi', 'Akbari',
                'Rajabi', 'Ebrahimi', 'Majidi', 'Mousavi', 'Hashemi', 'Rezaei',
                'Fadel', 'Khalil', 'Mansouri', 'Nouri', 'Rahimi', 'Sharif',
                'Tabatabaei', 'Yazdi', 'Zargham', 'Behzadi', 'Ghorbani', 'Shafiei'
            ],
            'African': [
                'Adebayo', 'Okafor', 'Ibekwe', 'Ogbonna', 'Eze', 'Emeka', 'Chukwu',
                'Okoro', 'Nwachukwu', 'Obinna', 'Kone', 'Traore', 'Diallo', 'Bah',
                'Camara', 'Soumah', 'Barry', 'Fofana', 'Sangare', 'Sissoko',
                'Mthembu', 'Ndlovu', 'Nkomo', 'Dube', 'Moyo', 'Sibanda', 'Ncube',
                'Mathe', 'Khumalo', 'Mpofu', 'Asante', 'Boateng', 'Owusu', 'Mensah',
                'Appiah', 'Osei', 'Adjei', 'Agyei', 'Gyasi', 'Ofori', 'Ansah',
                'Mugabe', 'Chivhayo', 'Mutasa', 'Mukamuri', 'Chidzonga', 'Makoni',
                'Mandaza', 'Nyagumbo', 'Zvobgo', 'Matongo', 'Muchena', 'Chinamasa',
                'Abebe', 'Tadesse', 'Bekele', 'Haile', 'Kebede', 'Desta', 'Tesfaye',
                'Alemayehu', 'Worku', 'Assefa', 'Getachew', 'Mulugeta', 'Kassahun',
                'Mwangi', 'Kamau', 'Njoroge', 'Wanjiku', 'Kiprotich', 'Cheruiyot',
                'Rotich', 'Koech', 'Biwott', 'Lagat', 'Kipchoge', 'Kemboi'
            ],
            'European': [
                'Müller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Meyer',
                'Wagner', 'Becker', 'Schulz', 'Hoffmann', 'Schäfer', 'Koch',
                'Bauer', 'Richter', 'Klein', 'Wolf', 'Schröder', 'Neumann',
                'Rossi', 'Russo', 'Ferrari', 'Esposito', 'Bianchi', 'Romano',
                'Colombo', 'Ricci', 'Marino', 'Greco', 'Bruno', 'Gallo',
                'Conti', 'De Luca', 'Mancini', 'Costa', 'Giordano', 'Rizzo',
                'Andersson', 'Johansson', 'Karlsson', 'Nilsson', 'Eriksson',
                'Larsson', 'Olsson', 'Persson', 'Svensson', 'Gustafsson',
                'Pettersson', 'Jonsson', 'Jansson', 'Hansson', 'Bengtsson',
                'Dupont', 'Martin', 'Bernard', 'Thomas', 'Petit', 'Robert',
                'Richard', 'Durand', 'Dubois', 'Moreau', 'Laurent', 'Simon',
                'Michel', 'Lefebvre', 'Leroy', 'Roux', 'David', 'Bertrand',
                'Popov', 'Petrov', 'Ivanov', 'Smirnov', 'Volkov', 'Sokolov',
                'Mikhailov', 'Fedorov', 'Morozov', 'Bogdanov', 'Orlov', 'Kiselev'
            ],
            'South Asian': [
                'Sharma', 'Verma', 'Gupta', 'Agarwal', 'Bansal', 'Jain', 'Singh',
                'Kumar', 'Yadav', 'Tiwari', 'Mishra', 'Shukla', 'Pandey', 'Srivastava',
                'Tripathi', 'Dwivedi', 'Upadhyay', 'Joshi', 'Bhatt', 'Saxena',
                'Khan', 'Sheikh', 'Malik', 'Hussain', 'Chaudhry', 'Butt', 'Awan',
                'Bajwa', 'Mughal', 'Qureshi', 'Syed', 'Shah', 'Mirza', 'Bhutto',
                'Patel', 'Shah', 'Gandhi', 'Modi', 'Thakkar', 'Vyas', 'Trivedi',
                'Parekh', 'Parikh', 'Desai', 'Joshi', 'Mehta', 'Raval', 'Amin',
                'Perera', 'Silva', 'Fernando', 'Rodrigo', 'Jayawardena', 'Kumara',
                'Wickramasinghe', 'Mendis', 'Gunasekara', 'Amarasinghe', 'Rajapaksa',
                'Bandaranaike', 'Senanayake', 'Jayasuriya', 'Dissanayake', 'Wijeratne',
                'Rahman', 'Ahmed', 'Hassan', 'Ali', 'Hossain', 'Islam', 'Khan',
                'Chowdhury', 'Roy', 'Das', 'Biswas', 'Chakraborty', 'Sarkar'
            ],
            'Indigenous': [
                'Crow Feather', 'Running Bear', 'White Eagle', 'Storm Cloud', 'Red Hawk',
                'Silent Wolf', 'Thunder Horse', 'Morning Star', 'Night Wind', 'Golden Eagle',
                'Brave Wolf', 'Singing Bird', 'Dancing Bear', 'Swift Arrow', 'Proud Elk',
                'Fire Horse', 'Wild Buffalo', 'Gentle Rain', 'Strong Bear', 'Flying Eagle',
                'Spotted Horse', 'Lone Wolf', 'Gray Fox', 'Black Hawk', 'White Wolf',
                'Tall Tree', 'Fast River', 'Bright Moon', 'Dark Cloud', 'Wise Owl',
                'Yazzie', 'Begay', 'Tsosie', 'Johnson', 'Benally', 'Smith', 'Jones',
                'Brown', 'Charlie', 'Willie', 'Garcia', 'Martinez', 'Lee', 'Wilson',
                'Etsitty', 'Nez', 'Chee', 'Blackhorse', 'Silversmith', 'Antonio',
                'Nakai', 'Todacheenie', 'Largo', 'Holiday', 'Clitso', 'Goldtooth',
                'Begaye', 'Platero', 'Shorty', 'Tso', 'Charley', 'Thompson',
                'Yellowhorse', 'Jim', 'Francisco', 'Montoya', 'Sandoval', 'Chavez'
            ],
            'Mixed/International': [
                'Anderson', 'Peterson', 'Thompson', 'Williams', 'Johnson', 'Brown',
                'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Jackson', 'White',
                'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson',
                'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen',
                'Young', 'Hernandez', 'King', 'Wright', 'Lopez', 'Hill', 'Scott',
                'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 'Carter', 'Mitchell',
                'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker',
                'Evans', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris',
                'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey',
                'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres',
                'Peterson', 'Gray', 'Ramirez', 'James', 'Watson', 'Brooks', 'Kelly',
                'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes', 'Ross', 'Henderson'
            ]
        }

        # Usage tracking for anti-bias
        self.name_usage_counter = Counter()
        self.role_tracking = defaultdict(lambda: defaultdict(int))
        self.max_usage_per_name = 50  # Prevent overuse like "Wounded Hawk" (723 times)

        # Cultural consistency weights
        self.culture_match_weight = 0.85  # 85% chance first name matches surname culture
        self.title_weights = {
            'gendered': 0.7,      # 70% chance of gendered title
            'professional': 0.25,  # 25% chance of professional title
            'none': 0.05          # 5% chance of no title
        }

        # Initialize statistical tracking
        self.generation_stats = {
            'total_generated': 0,
            'cultures_used': Counter(),
            'genders_used': Counter(),
            'titles_used': Counter(),
            'culture_mismatches': 0
        }

        logger.info("NameInventory initialized with comprehensive cultural databases")

    def generate_name(
        self,
        gender: Optional[str] = None,
        culture: Optional[str] = None,
        force_title: Optional[str] = None,
        avoid_overused: bool = True
    ) -> Dict[str, str]:
        """
        Generate a culturally consistent and diverse name.

        Args:
            gender: 'male', 'female', or None for random
            culture: Specific culture or None for random
            force_title: Specific title or None for weighted selection
            avoid_overused: Skip names that have been used too frequently

        Returns:
            Dict with keys: 'full_name', 'title', 'first_name', 'surname',
                          'gender', 'culture', 'usage_count'
        """

        max_attempts = 100  # Prevent infinite loops
        attempt = 0

        while attempt < max_attempts:
            # Select gender
            if gender is None:
                gender = random.choice(['male', 'female'])

            # Select culture for surname (this determines the base culture)
            if culture is None:
                surname_culture = random.choice(list(self.surnames.keys()))
            else:
                surname_culture = culture

            # Select first name culture (85% match, 15% mixed)
            if random.random() <= self.culture_match_weight:
                firstname_culture = surname_culture
            else:
                # 15% chance of cultural mixing
                firstname_culture = random.choice(list(self.first_names.keys()))
                self.generation_stats['culture_mismatches'] += 1

            # Generate name components
            surname = random.choice(self.surnames[surname_culture])

            # Ensure the gender exists in the culture
            available_genders = list(self.first_names[firstname_culture].keys())
            if gender not in available_genders:
                gender = random.choice(available_genders)

            first_name = random.choice(self.first_names[firstname_culture][gender])

            # Generate title
            if force_title:
                title = force_title
            else:
                title = self._select_title(gender)

            # Construct full name
            if title:
                full_name = f"{title} {first_name} {surname}"
            else:
                full_name = f"{first_name} {surname}"

            # Check usage frequency (anti-bias mechanism)
            if avoid_overused and self.name_usage_counter[full_name] >= self.max_usage_per_name:
                attempt += 1
                continue

            # Update tracking
            self.name_usage_counter[full_name] += 1
            self.generation_stats['total_generated'] += 1
            self.generation_stats['cultures_used'][surname_culture] += 1
            self.generation_stats['genders_used'][gender] += 1
            self.generation_stats['titles_used'][title if title else 'none'] += 1

            return {
                'full_name': full_name,
                'title': title,
                'first_name': first_name,
                'surname': surname,
                'gender': gender,
                'culture': surname_culture,
                'usage_count': self.name_usage_counter[full_name],
                'firstname_culture': firstname_culture,
                'culture_mixed': firstname_culture != surname_culture
            }

        # If we couldn't generate a non-overused name, generate anyway with warning
        logger.warning(f"Could not generate non-overused name after {max_attempts} attempts")
        result = {
            'full_name': full_name,
            'title': title,
            'first_name': first_name,
            'surname': surname,
            'gender': gender,
            'culture': surname_culture,
            'usage_count': self.name_usage_counter[full_name],
            'firstname_culture': firstname_culture,
            'culture_mixed': firstname_culture != surname_culture
        }
        self.name_usage_counter[full_name] += 1
        return result

    def _select_title(self, gender: str) -> Optional[str]:
        """Select a title based on weighted probabilities."""
        rand = random.random()

        if rand <= self.title_weights['gendered']:
            # Gendered title
            if gender in self.titles['gendered']:
                return random.choice(self.titles['gendered'][gender])
            else:
                return random.choice(self.titles['gendered']['neutral'])
        elif rand <= self.title_weights['gendered'] + self.title_weights['professional']:
            # Professional title
            return random.choice(self.titles['professional'])
        else:
            # No title
            return None

    def generate_batch(
        self,
        count: int,
        gender_distribution: Optional[Dict[str, float]] = None,
        culture_distribution: Optional[Dict[str, float]] = None,
        avoid_overused: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generate a batch of diverse names with specified distributions.

        Args:
            count: Number of names to generate
            gender_distribution: {'male': 0.45, 'female': 0.55} or None for equal
            culture_distribution: Culture weights or None for equal
            avoid_overused: Skip overused names

        Returns:
            List of name dictionaries
        """

        # Set default distributions
        if gender_distribution is None:
            gender_distribution = {'male': 0.5, 'female': 0.5}

        if culture_distribution is None:
            cultures = list(self.surnames.keys())
            weight = 1.0 / len(cultures)
            culture_distribution = {culture: weight for culture in cultures}

        # Generate weighted selections
        names = []
        for _ in range(count):
            # Select gender based on distribution
            gender = self._weighted_choice(gender_distribution)

            # Select culture based on distribution
            culture = self._weighted_choice(culture_distribution)

            name = self.generate_name(
                gender=gender,
                culture=culture,
                avoid_overused=avoid_overused
            )
            names.append(name)

        return names

    def _weighted_choice(self, distribution: Dict[str, float]) -> str:
        """Select an item based on weighted distribution."""
        rand = random.random()
        cumulative = 0.0

        for item, weight in distribution.items():
            cumulative += weight
            if rand <= cumulative:
                return item

        # Fallback to last item
        return list(distribution.keys())[-1]

    def track_role_assignment(self, name: str, role: str):
        """Track role assignments to detect bias patterns."""
        self.role_tracking[name][role] += 1

    def get_bias_report(self) -> Dict:
        """
        Generate a comprehensive bias analysis report.

        Returns:
            Dictionary with bias metrics and overused names
        """

        # Identify severely biased names (>80% one role)
        biased_names = {}
        for name, roles in self.role_tracking.items():
            if sum(roles.values()) >= 5:  # Only names used 5+ times
                total = sum(roles.values())
                max_role = max(roles.keys(), key=lambda r: roles[r])
                max_percentage = (roles[max_role] / total) * 100

                if max_percentage > 80:
                    biased_names[name] = {
                        'dominant_role': max_role,
                        'percentage': max_percentage,
                        'total_uses': total,
                        'role_distribution': dict(roles)
                    }

        # Identify overused names
        overused_names = {
            name: count for name, count in self.name_usage_counter.items()
            if count > self.max_usage_per_name
        }

        # Cultural diversity analysis
        total_names = self.generation_stats['total_generated']
        culture_percentages = {
            culture: (count / total_names) * 100 if total_names > 0 else 0
            for culture, count in self.generation_stats['cultures_used'].items()
        }

        return {
            'total_names_generated': total_names,
            'unique_names': len(self.name_usage_counter),
            'average_uses_per_name': total_names / len(self.name_usage_counter) if self.name_usage_counter else 0,
            'overused_names': overused_names,
            'severely_biased_names': biased_names,
            'culture_distribution': culture_percentages,
            'culture_mismatches': self.generation_stats['culture_mismatches'],
            'generation_stats': dict(self.generation_stats)
        }

    def reset_usage_tracking(self):
        """Reset all usage counters for fresh generation."""
        self.name_usage_counter.clear()
        self.role_tracking.clear()
        self.generation_stats = {
            'total_generated': 0,
            'cultures_used': Counter(),
            'genders_used': Counter(),
            'titles_used': Counter(),
            'culture_mismatches': 0
        }
        logger.info("Usage tracking reset")

    def export_inventory(self, filepath: str, format: str = 'json'):
        """
        Export the complete name inventory to file.

        Args:
            filepath: Output file path
            format: 'json' or 'pickle'
        """

        export_data = {
            'titles': self.titles,
            'first_names': self.first_names,
            'surnames': self.surnames,
            'name_usage_counter': dict(self.name_usage_counter),
            'role_tracking': {name: dict(roles) for name, roles in self.role_tracking.items()},
            'generation_stats': dict(self.generation_stats),
            'max_usage_per_name': self.max_usage_per_name,
            'culture_match_weight': self.culture_match_weight,
            'title_weights': self.title_weights
        }

        filepath = Path(filepath)

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Name inventory exported to {filepath}")

    def load_inventory(self, filepath: str, format: str = 'json'):
        """
        Load name inventory from file.

        Args:
            filepath: Input file path
            format: 'json' or 'pickle'
        """

        filepath = Path(filepath)

        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Load data into instance
        self.titles = data['titles']
        self.first_names = data['first_names']
        self.surnames = data['surnames']
        self.name_usage_counter = Counter(data['name_usage_counter'])
        self.role_tracking = defaultdict(lambda: defaultdict(int))
        for name, roles in data['role_tracking'].items():
            for role, count in roles.items():
                self.role_tracking[name][role] = count
        self.generation_stats = data['generation_stats']
        self.max_usage_per_name = data['max_usage_per_name']
        self.culture_match_weight = data['culture_match_weight']
        self.title_weights = data['title_weights']

        logger.info(f"Name inventory loaded from {filepath}")

    def get_inventory_stats(self) -> Dict:
        """Get comprehensive statistics about the name inventory."""

        # Calculate total possible combinations
        total_combinations = 0
        for culture, names_by_gender in self.first_names.items():
            for gender, first_names in names_by_gender.items():
                if culture in self.surnames:
                    total_combinations += len(first_names) * len(self.surnames[culture])

        # Title variations
        total_titles = (
            len(self.titles['professional']) +
            sum(len(titles) for titles in self.titles['gendered'].values())
        )

        return {
            'cultures_available': list(self.surnames.keys()),
            'total_cultures': len(self.surnames),
            'total_first_names': sum(
                len(names) for culture_names in self.first_names.values()
                for names in culture_names.values()
            ),
            'total_surnames': sum(len(surnames) for surnames in self.surnames.values()),
            'total_titles': total_titles,
            'estimated_unique_combinations': total_combinations * (total_titles + 1),  # +1 for no title
            'first_names_by_culture': {
                culture: {
                    gender: len(names) for gender, names in gender_names.items()
                } for culture, gender_names in self.first_names.items()
            },
            'surnames_by_culture': {
                culture: len(surnames) for culture, surnames in self.surnames.items()
            }
        }


def main():
    """Demonstration and testing of the NameInventory system."""

    print("🏗️  Initializing Name Inventory System...")
    inventory = NameInventory(seed=42)

    # Show inventory statistics
    stats = inventory.get_inventory_stats()
    print(f"\n📊 INVENTORY STATISTICS:")
    print(f"   Cultures: {stats['total_cultures']}")
    print(f"   First names: {stats['total_first_names']:,}")
    print(f"   Surnames: {stats['total_surnames']:,}")
    print(f"   Titles: {stats['total_titles']}")
    print(f"   Estimated combinations: {stats['estimated_unique_combinations']:,}")

    # Generate sample names
    print(f"\n🎭 SAMPLE NAME GENERATION:")
    for i in range(15):
        name = inventory.generate_name()
        print(f"   {name['full_name']:<25} [{name['culture']}, {name['gender']}, mixed: {name['culture_mixed']}]")

    # Generate cultural batch
    print(f"\n🌍 CULTURAL DISTRIBUTION TEST:")
    batch = inventory.generate_batch(
        count=100,
        culture_distribution={
            'Anglo/Western': 0.3,
            'Hispanic/Latino': 0.2,
            'East Asian': 0.15,
            'African': 0.15,
            'Middle Eastern/Arabic': 0.1,
            'European': 0.05,
            'South Asian': 0.03,
            'Indigenous': 0.02
        }
    )

    # Analyze batch diversity
    cultures = Counter(name['culture'] for name in batch)
    genders = Counter(name['gender'] for name in batch)
    mixed_cultures = sum(1 for name in batch if name['culture_mixed'])

    print(f"   Generated {len(batch)} names:")
    for culture, count in cultures.most_common():
        percentage = (count / len(batch)) * 100
        print(f"     {culture:<20} {count:>3} ({percentage:>5.1f}%)")

    print(f"   Gender distribution: {dict(genders)}")
    print(f"   Cultural mixing: {mixed_cultures} ({mixed_cultures/len(batch)*100:.1f}%)")

    # Export sample inventory
    print(f"\n💾 EXPORTING SAMPLE INVENTORY...")
    inventory.export_inventory('/home/mitch/rune-lib/sample_name_inventory.json')

    print(f"\n✅ Name Inventory System demonstration complete!")
    print(f"   Ready for integration with NER training data augmentation")


if __name__ == "__main__":
    main()