/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov
 *  
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 * Alexey N. Salnikov salnikov@cmc.msu.ru
 *
 */

import parus.data.Network_data;
import parus.win.Network_view_window;
import parus.common.Objects_loader;

import java.io.File;
import java.io.IOException;

import java.net.URL;

import javax.swing.JFrame;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JButton;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.ImageIcon;
import javax.swing.JTextField;
import javax.swing.JOptionPane;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.Container;
import java.awt.BorderLayout;
import java.awt.GridLayout;



public class Network_viewer
{
	private JFrame frame;
	private JLabel label;
	private int default_data_window_size;
	
	private class Open_action implements ActionListener
	{
		private File data_file=null;
		private File deviation_file=null;

		JOptionPane pane;

		public void actionPerformed(ActionEvent e)
		{
			JFileChooser chooser = new JFileChooser(".");
		

			boolean open_deviation_flag=false;
			boolean data_file_chosen=false;
						
			pane=new JOptionPane
			(
				"Load data from deviation file?",
				JOptionPane.QUESTION_MESSAGE,
				JOptionPane.YES_OPTION
			);

			JDialog d=pane.createDialog(frame,"Deviation file confirmation");
			d.setVisible(true);

			Object value=pane.getValue();
			if(value!=null)
			{
				if((Integer)value==0)
				{
					open_deviation_flag=true;
				}
			}
			pane.setValue(null);
			System.out.println(pane.getValue());

			int returnVal = chooser.showDialog(frame,"Open data file");
			if(returnVal == JFileChooser.APPROVE_OPTION)
			{
				data_file=chooser.getSelectedFile();
				data_file_chosen=true;
			
			}
			
			if(open_deviation_flag)
			{
				returnVal = chooser.showDialog(frame,"Open deviation file");
				if(returnVal == JFileChooser.APPROVE_OPTION)
				{
					deviation_file=chooser.getSelectedFile();
				}
			}
			
			if(data_file_chosen)
			{
				Opener_data opener=new Opener_data(data_file,deviation_file);
				data_file=null;
				deviation_file=null;
				Thread thread=new Thread(opener);
				thread.start();
			}


		}

	}

	private class Data_window_size_action implements ActionListener
	{
		JDialog dialog;
		JTextField text_field;

		private class Butt_acct implements ActionListener
		{
			public void actionPerformed(ActionEvent e)
			{
				String str;
				str=text_field.getText();
				Integer ival=new Integer(str);
				default_data_window_size=ival.intValue();
				dialog.setVisible(false);
			}
		}

		public void actionPerformed(ActionEvent e)
		{
			String str=new String();
			dialog=new JDialog(frame,"window size",true);

			Container cpane = dialog.getContentPane();

			text_field=new JTextField(str+default_data_window_size);
			cpane.setLayout(new GridLayout(1,2));			
			
			JButton butt=new JButton("set window size");
			butt.addActionListener(new Butt_acct());

			cpane.add(text_field);
			cpane.add(butt);
			


			dialog.setSize(200,25);
			dialog.setLocationByPlatform(true);
			dialog.pack();
			dialog.setVisible(true);
			dialog.setFocusable(true);
			dialog.repaint();
		}


	}
	
	private class Exit_action implements ActionListener
	{
	
		public void actionPerformed(ActionEvent e)
		{
			System.exit(0);
		}
	}

	private class Opener_data implements Runnable
	{
		File data_file;
		File deviation_file;

		public Opener_data(File f,File d)
		{
			data_file=f;
			deviation_file=d;
		}

		public void run()
		{
			String file_name;
			
			Network_data deviation=null;
			Network_data data;

			file_name=data_file.getName();	
			
			try
			{
					label.setText("Open file with name '"+file_name+"'");
					label.repaint();

					data = new Network_data(data_file,default_data_window_size);
					//System.out.println(data.toString());
			}
			catch (IOException ex)
			{
				System.out.println("Read data from "+file_name+" failed");
				System.out.println(ex.getMessage());
				
				label.setText("File with name '"+file_name+"' can't be open!");
				label.repaint();
				
				return;
			}
			//System.out.println(data.toString());
				
			label.setText("File with name '"+file_name+"' opened");
			label.repaint();


			try
			{
										
					if(deviation_file!=null)
					{
						file_name=deviation_file.getName();
						deviation=new Network_data(deviation_file,default_data_window_size);
					}
				
					//System.out.println(data.toString());
			}
			catch (IOException ex)
			{
				System.out.println("Read deviation from "+file_name+" failed");
				System.out.println(ex.getMessage());
				
				label.setText("File with name '"+file_name+"' can't be open!");
				label.repaint();
				
				return;
			}
			
			label.setText("File with name '"+file_name+"' opened");
			label.repaint();

			Network_view_window window = new Network_view_window(frame,data,deviation);

		
		}
	}
 
	
	public static void main(String args[]) throws java.io.IOException
	{
		
		Network_viewer network_viewer;
		if(args.length > 0)
		{
			network_viewer=new Network_viewer(args[0]);
			System.out.println("Ok");
		}
		else
		{
			network_viewer=new Network_viewer(null);
		}
	}

	Network_viewer(String file_name) throws java.io.IOException
	{
		default_data_window_size=20;
		
		frame = new JFrame("Viewer of network test results");
		Container pane = frame.getContentPane();

		/*
		 * Menu building 
		 */
		JMenuBar   menu_bar       = new JMenuBar();
		JMenu      file_menu      = new JMenu("File");
		JMenuItem  open_menu_item = new JMenuItem("Open");
		JMenuItem  exit_menu_item = new JMenuItem("Exit");

		JMenu      options_menu   = new JMenu("Options");
		JMenuItem  data_window_size_menu_item = new JMenuItem("Window size");

			
		open_menu_item.addActionListener(new Open_action());
		exit_menu_item.addActionListener(new Exit_action());

		data_window_size_menu_item.addActionListener(new Data_window_size_action());

		file_menu.add(open_menu_item);
		file_menu.add(exit_menu_item);

		options_menu.add(data_window_size_menu_item);

		menu_bar.add(file_menu);
		menu_bar.add(options_menu);

		frame.setJMenuBar(menu_bar);
		frame.setLayout(new BorderLayout());
		
		URL path=Objects_loader.get_file("images/logotype.png");
		ImageIcon icon=new ImageIcon(path);
		JLabel image_label= new JLabel(icon);
		
		/*
		 * Surface space building
		 */
		
		JPanel panel=new JPanel();
		panel.setLayout(new GridLayout(1,1));
		panel.add(image_label);
		
		pane.add(BorderLayout.CENTER,panel);
		label= new JLabel("");
		pane.add(BorderLayout.SOUTH,label);
		
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(250,250);
		frame.setLocationByPlatform(true);
		frame.setIconImage(icon.getImage());
		frame.setVisible(true);
		
		if(file_name!=null)
		{
			File file=new File(file_name);
			
			Opener_data opener=new Opener_data(file,null);
			Thread thread=new Thread(opener);
			thread.start();

		}		
	}
}

